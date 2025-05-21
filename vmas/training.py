import torch
import torch.nn as nn
from torchrl.envs.libs.vmas import VmasWrapper
from tensordict import TensorDict
from vmas import make_env as vmas_make_env
from policy import HighLevelPolicy
from vmas.scenarios.football import AgentPolicy
from critic import CentralCritic
import matplotlib

# Set matplotlib to non-interactive backend for headless plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Hyperparameters ---
num_envs = 1
num_steps = 128
n_agents = 2
hidden_dim = 128
lr = 1e-3

# === Environment Setup ===
raw_env = vmas_make_env(
    scenario="football",
    num_envs=num_envs,
    n_blue_agents=n_agents,
    n_red_agents=1,
    ai_blue_agents=False,
    ai_red_agents=True,
    disable_ai_red=True,
    continuous_actions=True,
    dict_obs=False,
    observe_teammates=False,
    device="cpu",
)
env = VmasWrapper(env=raw_env)
print("Environment created.")

# Scenario and low-level policy setup
scenario = raw_env.scenario
low_level = AgentPolicy(team="Blue")
low_level.init(scenario.world)
low_level.plot_traj = lambda *args, **kwargs: None  # Disable plotting

# --- Inference of Observation/Action Space ---
td = env.reset()
obs = td["agent_blue"]["observation"]  # [batch, n_agents, obs_dim]
print(obs)
print(obs.shape)
obs_dim = obs.shape[-1]
action_dim = env.action_spec["agent_blue"]["action"].shape[-1]

# --- Initialize Policy and Critic Networks ---
policy = HighLevelPolicy(obs_dim, n_agents, hidden_dim)
critic = CentralCritic(total_obs_dim=obs_dim * n_agents, n_agents=n_agents)

# --- Data Collection (Rollout) ---
rollout = []
td = env.reset()
print("✅ Environment reset.")
reward_per_step = []

for step in range(num_steps):
    obs = td["agent_blue"]["observation"]  # [num_envs, n_agents, obs_dim]
    obs_flat = obs.view(num_envs * n_agents, obs_dim)  # Flatten for policy

    with torch.no_grad():
        mode, params, logp_mode = policy(obs_flat)
        mode = mode.view(num_envs, n_agents)
        params = params.view(num_envs, n_agents, -1)
        logp_mode = logp_mode.view(num_envs, n_agents, 1)
        log_prob = logp_mode

        # Vectorized low-level actions for all agents
        low_actions = torch.zeros(num_envs, n_agents, action_dim, device=obs.device)
        ball_pos = scenario.ball.state.pos
        goal_pos = scenario.right_goal_pos

        for i, agent in enumerate(scenario.blue_agents):
            m_i = mode[:, i]
            p_i = params[:, i, :]
            tp = torch.zeros_like(ball_pos)
            tv = torch.zeros_like(ball_pos)

            mask = (m_i == 0)  # Move to ball
            tp[mask] = ball_pos[mask]
            tv[mask] = 0.0

            mask = (m_i == 1)  # Dribble
            d = p_i[mask, :2]
            norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            tp[mask] = ball_pos[mask] + d / norm
            tv[mask] = d

            mask = (m_i == 2)  # Shoot
            tp[mask] = goal_pos.unsqueeze(0).expand_as(tp)[mask]
            power = torch.sigmoid(p_i[mask, 2:3])
            dir_vec = (tp[mask] - ball_pos[mask])
            dir_norm = dir_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            tv[mask] = dir_vec / dir_norm * power

            mask = (m_i == 3)  # Pass
            if mask.any():
                idx = p_i[mask, 3:].argmax(dim=-1)
                for k, envi in enumerate(mask.nonzero(as_tuple=False).squeeze(-1)):
                    j = idx[k].item()
                    tpos = scenario.blue_agents[j].state.pos[envi]
                    tp[envi] = tpos
                    tv[envi] = tpos - ball_pos[envi]

            low_level.go_to(agent, pos=tp, vel=tv, env_index=Ellipsis)
            u = low_level.get_action(agent, env_index=Ellipsis)
            u = u.clamp(min=-agent.u_range, max=agent.u_range)
            mult = agent.action.u_multiplier_tensor.unsqueeze(0)
            u = u * mult
            low_actions[:, i, :] = u

        # Environment step
        action_dict = TensorDict({"action": low_actions}, batch_size=[num_envs, n_agents])
        td = env.step(TensorDict({"agent_blue": action_dict}, batch_size=[num_envs]))["next"]

    reward = td["agent_blue"]["reward"]
    reward_per_step.append(reward.mean().item())
    done = td["done"]

    rollout.append({
        "obs": obs.clone(),
        "mode": mode.clone(),
        "params": params.clone(),
        "log_prob": log_prob.clone(),
        "reward": reward.clone(),
        "done": done.clone(),
    })

    if done.any():
        td = env.reset()

print(f"\nTotal steps collected: {len(rollout)}")


# --- Return Calculation ---
def compute_returns(rews, gamma=0.99):
    """Compute discounted returns for each timestep."""
    T, N, _ = rews.shape
    R = torch.zeros_like(rews)
    G = torch.zeros(N, 1)
    for t in reversed(range(T)):
        G = rews[t] + gamma * G
        R[t] = G
    return R


# --- Prepare Training Data ---
obs = torch.cat([t["obs"] for t in rollout], dim=0)
logp = torch.cat([t["log_prob"] for t in rollout], dim=0)
rews = torch.cat([t["reward"] for t in rollout], dim=0)
rets = compute_returns(rews)
T, N, _ = rews.shape
obs_flat = obs.view(T, N, -1)
joint_obs = obs_flat.view(T, -1)
returns_flat = rets.view(T, N)

with torch.no_grad():
    vals = critic(joint_obs)

advs = returns_flat - vals

train_batch = {
    "obs": obs.view(T * N, -1),
    "log_prob": logp.view(T * N, 1),
    "returns": returns_flat.view(T * N, 1),
    "advantages": advs.view(T * N, 1),
    "joint_obs": joint_obs
}

# --- PPO Training Loop ---
policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
clip_eps = 0.2
value_coef = 0.5
entropy_coef = 0.01
n_epochs = 4
batch_size = 32
dataset_size = train_batch["obs"].shape[0]

for epoch in range(n_epochs):
    idx = torch.randperm(dataset_size)
    for start in range(0, dataset_size, batch_size):
        batch = idx[start:start + batch_size]
        b_obs = train_batch["obs"][batch]
        b_logp = train_batch["log_prob"][batch]
        b_adv = train_batch["advantages"][batch]
        b_ret = train_batch["returns"][batch]
        batch_timesteps = batch // n_agents
        batch_agent_ids = batch % n_agents
        b_joint = train_batch["joint_obs"][batch_timesteps]
        mode_n, params_n, logp_n = policy(b_obs)
        ratio = torch.exp(logp_n - b_logp)
        s1 = ratio * b_adv
        s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
        policy_loss = -torch.min(s1, s2).mean()
        entropy_loss = -logp_n.mean()
        vals = critic(b_joint)
        v_pred = vals.gather(1, batch_agent_ids.unsqueeze(1))
        value_loss = (v_pred - b_ret).pow(2).mean()
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        policy_optim.zero_grad()
        critic_optim.zero_grad()
        loss.backward()
        policy_optim.step()
        critic_optim.step()

print("PPO update completed.")


# --- Evaluation/Rendering ---
def render_episode(policy):
    """Evaluate the trained policy in a single episode with rendering."""
    render_env = VmasWrapper(vmas_make_env(
        scenario="football",
        num_envs=num_envs,
        n_blue_agents=n_agents,
        n_red_agents=1,
        ai_blue_agents=False,
        ai_red_agents=True,
        disable_ai_red=True,
        continuous_actions=True,
        observe_teammates=False,
        device="cpu",
    ))
    td = render_env.reset()
    raw2 = render_env._env
    scenario = raw2.scenario
    low_level.init(scenario.world)
    for _ in range(300):
        obs = td["agent_blue"]["observation"].squeeze(0)  # [1, A, D] → [A, D]
        obs_flat = obs.view(n_agents, -1)
        with torch.no_grad():
            mode, params, _ = policy(obs_flat)

        low_actions = torch.zeros(1, n_agents, action_dim, device=obs.device)
        for i in range(n_agents):
            m = mode[i].item()
            p = params[i]

            if m == 0:  # move to ball
                target_pos = scenario.ball.state.pos[0]
                target_vel = torch.zeros_like(target_pos)
            elif m == 1:  # dribble
                d = p[:2]
                target_pos = scenario.ball.state.pos[0] + d / (d.norm() + 1e-8)
                target_vel = d
            elif m == 2:  # shoot
                target_pos = scenario.right_goal_pos
                power = torch.sigmoid(p[2])
                dir = target_pos - scenario.ball.state.pos[0]
                target_vel = dir / (dir.norm() + 1e-8) * power
            else:  # pass
                j = p[3:].argmax().item()
                tp = scenario.blue_agents[j].state.pos[0]
                target_pos = tp
                target_vel = tp - scenario.ball.state.pos[0]

            low_level.go_to(
                scenario.blue_agents[i],
                pos=target_pos.unsqueeze(0),
                vel=target_vel.unsqueeze(0),
                env_index=Ellipsis
            )

            raw_u = low_level.get_action(
                scenario.blue_agents[i],
                env_index=Ellipsis
            )
            clamped = raw_u.clamp(min=-scenario.blue_agents[i].u_range,
                                  max=scenario.blue_agents[i].u_range)
            mult = scenario.blue_agents[i].action.u_multiplier_tensor.unsqueeze(0)
            scaled = clamped * mult
            low_actions[0, i] = scaled[0]

        td = render_env.step(
            TensorDict({"agent_blue":
                            TensorDict({"action": low_actions}, batch_size=[1, n_agents])},
                       batch_size=[1])
        )["next"]
        render_env.render()
        if td["done"].item():
            break


render_episode(policy)

# --- Reward Plot ---
plt.figure(figsize=(8, 4))
plt.plot(reward_per_step, label="Reward")
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig("reward_plot.png")
print("Wrote reward_plot.png")
