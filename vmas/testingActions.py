import torch
from torchrl.envs.libs.vmas import VmasWrapper
from tensordict import TensorDict
from vmas import make_env as vmas_make_env
from vmas.scenarios.football import AgentPolicy

def render_episode_basic():
    render_env = VmasWrapper(
        vmas_make_env(
            scenario="football",
            num_envs=1,
            n_blue_agents=1,
            n_red_agents=1,
            ai_blue_agents=False,
            ai_red_agents=True,
            disable_ai_red=True,
            continuous_actions=True,
            observe_teammates=False,
            device="cpu",
        )
    )

    td = render_env.reset()
    scenario = render_env._env.scenario
    agent = scenario.blue_agents[0]  # Only one agent
    low_level = AgentPolicy(team="Blue")
    low_level.init(scenario.world)
    low_level.plot_traj = lambda *args, **kwargs: None  # Disable plotting (caused errors)

    action_dim = render_env.action_spec["agent_blue"]["action"].shape[-1]
    print(f"action_dim = {action_dim}")

    DRIBBLE_THRESHOLD = 0.1
    SHOOT_THRESHOLD = 0.25  # Adjust as needed

    for _ in range(1000):
        agent_pos = agent.state.pos[0]
        ball_pos = scenario.ball.state.pos[0]
        goal_pos = scenario.right_goal_pos

        dist_to_ball = (agent_pos - ball_pos).norm()
        dist_ball_to_goal = (ball_pos - goal_pos).norm()

        for agent in scenario.blue_agents:
            # 1. Move to ball
            if dist_to_ball > DRIBBLE_THRESHOLD:
                print(f"Agent {agent.name} is MOVING TO BALL")
                low_level.go_to(agent, pos=ball_pos.unsqueeze(0))
                current_action = "move_to_ball"

            # 2. Dribble toward goal
            elif dist_ball_to_goal > SHOOT_THRESHOLD:
                print(f"Agent {agent.name} is DRIBBLING TOWARD GOAL")
                low_level.dribble_to_goal(agent)
                current_action = "dribble"

            # 3. Shoot if close enough
            else:
                print(f"Agent {agent.name} is SHOOTING!")
                low_level.shoot_at_goal(agent)
                current_action = "shoot"

            u = low_level.get_action(agent)[0]
            u = u.clamp(-1.0, 1.0)
            print(f"Action vector for {agent.name}: {u.cpu().numpy()} ({current_action})")

            # Add batch dimension and agent dimension
            actions = u.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, action_dim]

            td = render_env.step(
                TensorDict({"agent_blue":
                                TensorDict({"action": actions}, batch_size=[1])},
                           batch_size=[1])
            )["next"]
            render_env.render()
            if td["done"].item():
                td = render_env.reset()

if __name__ == "__main__":
    render_episode_basic()
