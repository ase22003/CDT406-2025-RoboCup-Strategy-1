#{{{PREAMBLE
#{{{MODULES
using Sockets
using Dates
#}}}
#{{{STRUCTURES
struct DataKey
	port::Int
	sock::UDPSocket
end
Base.:(==)(a::DataKey, b::DataKey) = a.id == b.id && a.socket === b.socket
Base.hash(k::DataKey, h::UInt) = hash(k.id, h) ⊻ objectid(k.socket)
mutable struct FunctionCall
	f::Function
	args::Tuple
end
#{{{FOOTBALL
mutable struct Point
	x::Float16
	y::Float16
end
mutable struct Velocity
	dx::Float16
	dy::Float16
end
mutable struct Physical
	position::Point
	velocity::Velocity
	angle::Float16
end
struct Player
	id::UInt8
	sock::UDPSocket
	name::String #team name
	physical::Physical
	info::Dict{String, Any} #roles are given by "role" herein
end
struct Team
	name::String
	side::Symbol
	players::Vector{Player}
end
#}}}
#}}}
#{{{UDP CONSTANTS
const IP = ip"127.0.0.1"
const PLAYER_PORT  = 6000
const TRAINER_PORT = 6001
const VERSION = 7
#}}}
#{{{SERVER COMMUNICATION DELAYS
const COMMAND_UPDATE_DELAY = 0.1 #command-sending delay (seconds)
const UPDATE_DELAY_MARGIN = 0.0
const WAIT_FOR_UPDATES = 0.1
#}}}
#{{{GLOBALS
const NUMBER_OF_PLAYERS = 6

global mutex_command_send = IdDict{DataKey, ReentrantLock}()
global field_state = Dict{String, Any}() #team 1, team 2, ball
global state_lock = ReentrantLock()
global GLOBAL_TRAINER_SOCKET = UDPSocket()
global agent_instructions_A = [FunctionCall(x->:x, (0,)) for i in 1:NUMBER_OF_PLAYERS]
global agent_instructions_B = [FunctionCall(x->:x, (0,)) for i in 1:NUMBER_OF_PLAYERS]

#define goal positions
field_state["goal"] = Dict{Symbol, Point}()
field_state["goal"][:RIGHT] = Point(52.5, 0)
field_state["goal"][:LEFT]  = Point(-52.5, 0)
#}}}
#}}}
#{{{FUNCTION DEFINITIONS
#{{{UDP PRIMITIVE
function Client(port::Int, server::String="localhost")::UDPSocket #initiate a client/socket (player)
    sock = UDPSocket()
    bind(sock, IP, 0)
    println("Connected to $server:$port")
	mutex_command_send[DataKey(port,sock)] = ReentrantLock() #add client to mutex dictionary
    sock
end

function send_command_primitive(port::Int, sock::UDPSocket, message::String)
	send(sock, IP, port, message)
	#println("$(time()) $port SENT \"$message\"")
end

function get_data_primitive(sock::UDPSocket)::String
	#println("@@@ WAITING... @@@")
	s=String(recv(sock))
	#println("GOT \"$s\"")
	s
end

function die(sock::UDPSocket)
    close(sock)
end
#}}}
#{{{CUSTOM COMMUNICATION PROTOCOLS
function send_command(port::Int, sock::UDPSocket, message::String)
	#println("$port:\"$message\"")
	Threads.@spawn begin
		lock(mutex_command_send[DataKey(port,sock)])
		send_command_primitive(port,sock,message)
		sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN) #wait so as to not send commands at the same time
		unlock(mutex_command_send[DataKey(port,sock)])
	end
end
function start_data_update_loop(sock::UDPSocket)
	Threads.@spawn begin
		while true
			send_command_primitive(TRAINER_PORT, sock, "(look)")
			sleep(COMMAND_UPDATE_DELAY)
			lock(state_lock) do
				data = get_data_primitive(sock)*" x x"
				if split(data, " ")[2] == "look"
					look_data_parse(data)
					#println("\t\tUPDATED: $(field_state)")
				end
			end
		end
	end
end
#{{{PARSE DATA
function look_data_parse(raw::String)
	raw = filter(!isempty, split(raw,('(',')',' ')))
	len = length(raw)
	if len < 2
		return :ERR
	end
	if raw[2] != "look"
		return :ERR
	end
	i = 1
	while true
		if i > len
			return
		end

		if raw[i] == "ball"
			global field_state["ball"] = 
				Physical(
						 Point(
							   parse(Float16, raw[i+1]),
						 	   parse(Float16, raw[i+2])),
						  Velocity(
								   parse(Float16, raw[i+3]),
						  		   parse(Float16, raw[i+4])),
						  Float16(0))
			i+=5
		elseif raw[i] == "player"
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.position.x  = parse(Float16, raw[i+3])
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.position.y  = -parse(Float16, raw[i+4])
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.velocity.dx = parse(Float16, raw[i+6])
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.velocity.dy = parse(Float16, raw[i+7])
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.angle       = parse(Float16, raw[i+5])
			i+=9
		else
			i+=1
		end
	end
end
#}}}
#}}}
#{{{TEAM INITIATION FUNCTIONS
function create_player(id::UInt8, name::String)::Player
	sock=Client(PLAYER_PORT)
	send_command(PLAYER_PORT, sock, "(init $name (version $VERSION))")
	info = Dict(); if id == 1 #the first player is always the goalie
		info["role"] = "goalie"
	end
	info["status"] = :undone
	Player(
		id,
		sock,
		name,
		Physical(Point(0, 0), Velocity(0, 0), 0),
		info
	)
end

function create_team(name::String, side::Symbol)::Team
	Team(
		name,
		side,
		Vector([create_player(UInt8(i), name) for i in 1:NUMBER_OF_PLAYERS])
	)
end
#}}}
#{{{LOW LEVEL SKILLS
#{{{REDUNDANT
#=
function PLAYER_goto(player::Player, position::Point, margin::UInt8, angle_precision::UInt8, break_distance::UInt8)
	Threads.@spawn begin
		while true
			Δx = player.physical.position.x - position.x
			Δy = player.physical.position.y + position.y
			dist = hypot(Δx, Δy) 
			#println("$(player.id): x=$(player.physical.position.x), y=$(player.physical.position.y), Δx=$Δx, Δy=$Δy, hyp=$dist")
			if dist < margin
				#println("$(player.id) END")
				return #tell the parent process?
			end
			
			θ = rad2deg(atan((position.y + player.physical.position.y), (position.x - player.physical.position.x)))
			pangle = -player.physical.angle
			Δθ = mod(pangle - θ + 180, 360) - 180
			if abs(Δθ) > angle_precision
				#send_command(PLAYER_PORT, player.port, "(turn $(Δθ/2))")
				#sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN+WAIT_FOR_UPDATES)
				send_command_primitive(PLAYER_PORT, player.port, "(turn $(Δθ))")
				sleep(COMMAND_UPDATE_DELAY)
			end
			#send_command(PLAYER_PORT, player.port, "(dash 100)")
			#sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN+WAIT_FOR_UPDATES)
			if dist < break_distance
				send_command_primitive(PLAYER_PORT, player.port, "(dash 20)")
			else
				send_command_primitive(PLAYER_PORT, player.port, "(dash 100)")
			end
			sleep(COMMAND_UPDATE_DELAY)
		end
	end
end
=#
#}}}
#all functions take two steps (0.2 seconds) and return either :done or :undone
#they are all aiming *toward* a goal, *not** to complete it
function PLAYER_idle(_)::Symbol
	:done
end
function PLAYER_go_toward(player::Player, position::Point, margin::Int, angle_precision::Int, break_distance::Int, break_amount::Int)::Symbol #go toward a particular coordinate
	tx = position.x #target position
	ty = position.y
	px = player.physical.position.x #current player position
	py = player.physical.position.y
	Δx = px - tx
	Δy = py - ty
	dist = hypot(Δx, Δy)
	#println("$(player.id): x=$(player.physical.position.x), y=$(player.physical.position.y), Δx=$Δx, Δy=$Δy, hyp=$dist")
	if dist < margin
		return :done #finished with task
	end
	
	θ = rad2deg(atan((ty - py), (tx - px)))
	pangle = -player.physical.angle
	Δθ = mod(pangle - θ + 180, 360) - 180

	#println("target: ($tx, $ty)")
	#println("$(player.id): x=$(px), y=$(py), Δx=$Δx, Δy=$Δy, hyp=$dist")
	if abs(Δθ) > angle_precision
		send_command_primitive(PLAYER_PORT, player.sock, "(turn $(Δθ/2))")
		sleep(COMMAND_UPDATE_DELAY)
		if dist < break_distance
			send_command_primitive(PLAYER_PORT, player.sock, "(dash 20)")
		else
			send_command_primitive(PLAYER_PORT, player.sock, "(dash 100)")
		end
	else
		if dist < break_distance
			send_command_primitive(PLAYER_PORT, player.sock, "(dash 20)")
		else
			send_command_primitive(PLAYER_PORT, player.sock, "(dash 100)")
		end
		sleep(COMMAND_UPDATE_DELAY)
		if dist < break_distance
			send_command_primitive(PLAYER_PORT, player.sock, "(dash 20)")
		else
			send_command_primitive(PLAYER_PORT, player.sock, "(dash 100)")
		end
	end
	:undone #not finished with task yet
end
function PLAYER_kick(player::Player, direction::Int, power::Int)
	tx = field_state["ball"].position.x #current ball position
	ty = field_state["ball"].position.y
	px = player.physical.position.x #current player position
	py = player.physical.position.y

	#θ = rad2deg(atan((ty - py), (tx - px)))
	θ = direction
	pangle = -player.physical.angle
	Δθ = mod(pangle - θ + 180, 360) - 180
	
	send_command_primitive(PLAYER_PORT, player.sock, "(turn $(Δθ))")
	sleep(COMMAND_UPDATE_DELAY)
	send_command_primitive(PLAYER_PORT, player.sock, "(kick $power $direction)")

	:done
end
#}}}
#{{{EXECUTOR
function executor(team::Team, agent_instructions::Vector)
	status = [:undone for i ∈ 1:NUMBER_OF_PLAYERS]
	while true
		for i=1:NUMBER_OF_PLAYERS
			Threads.@spawn begin
				try
					status[i] = agent_instructions[i].f(agent_instructions[i].args...)
				catch
					println("executor error")
				end
			end
			if team.players[i].info["status"] != status[i]
				lock(state_lock) do
					team.players[i].info["status"] = status[i]
					println("updated $i's status to $(status[i])")
				end
			end
		end
		sleep(COMMAND_UPDATE_DELAY*2)
	end
end
function update_player_instruction(team::Team, id::Int, func::Function, args::Tuple)
	if team.name == "Team_A"
		global agent_instructions_A[id] = FunctionCall(func,(team.players[id], args...))
	end
	if team.name == "Team_B"
		global agent_instructions_B[id] = FunctionCall(func,(team.players[id], args...))
	end
end
#}}}
#{{{DECISION MAKER

#}}}
#{{{MASTER
function master()
	teamnames = ("Team_A", "Team_B")

	#init trainer
	trainer = Client(TRAINER_PORT)
	GLOBAL_TRAINER_SOCKET = trainer
	send_command(TRAINER_PORT, trainer, "(init $(teamnames[1]) (version $VERSION))")
	start_data_update_loop(trainer)
	sleep(1)
	
	#init teams
	teams = (create_team(teamnames[1], :RIGHT),
			 create_team(teamnames[2], :LEFT))
	field_state[teamnames[1]] = teams[1]
	field_state[teamnames[2]] = teams[2]

	#global agent_instructions_A = [FunctionCall(PLAYER_idle, (0,)) for i in 1:NUMBER_OF_PLAYERS]
	#global agent_instructions_B = [FunctionCall(PLAYER_idle, (0,)) for i in 1:NUMBER_OF_PLAYERS]
	agent_instructions = (agent_instructions_A, agent_instructions_B)


	#define starting positions
	starting_positions = (
		Point(50,0), #goalie
		Point(15,20),
		Point(15,10),
		Point(15,0), ######### ADD/DEFINE "KICKER" POSITION
		Point(15,-10),
		Point(15,-20)
	)
	
	#go to starting positions
	#=
	side = Int8(-1)
	for j ∈ 1:2
		for i ∈ 1:NUMBER_OF_PLAYERS
				update_player_instruction(teams[j], i, PLAYER_go_toward, (Point(starting_positions[i].x*side, starting_positions[i].y), UInt8(1), UInt8(10), UInt8(3), UInt8(50)))
		end
		side = 1
	end
	=#
	

	#run game
	send_command(TRAINER_PORT, trainer, "(change_mode play_on)")
	
	Threads.@spawn executor(teams[1], agent_instructions_A)
	Threads.@spawn executor(teams[2], agent_instructions_B)



	update_player_instruction(teams[1], 1, PLAYER_go_toward, (field_state["ball"].position, 1, 15, 2, 50))
	sleep(1)
	while field_state[teamnames[1]].players[1].info["status"] == :undone
		sleep(COMMAND_UPDATE_DELAY)
	end
	update_player_instruction(teams[1], 1, PLAYER_kick, (0, 100))



	sleep(60)

	#kill game
	for team ∈ teams
		for player ∈ team.players
    		die(player.sock)
		end
	end
	die(trainer)
end
#}}}
#}}}
#{{{PROGRAM
master()
#}}}
