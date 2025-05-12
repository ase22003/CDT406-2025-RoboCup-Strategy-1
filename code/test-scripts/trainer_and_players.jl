#{{{PREAMBLE
#{{{MODULES
using Sockets
#}}}
#{{{STRUCTURES
struct DataKey
	port::Int
	sock::UDPSocket
end
Base.:(==)(a::DataKey, b::DataKey) = a.id == b.id && a.socket === b.socket
Base.hash(k::DataKey, h::UInt) = hash(k.id, h) ⊻ objectid(k.socket)
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
	port::UDPSocket
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
#}}}
#{{{GLOBALS
global mutex_command_send = IdDict{DataKey, ReentrantLock}()
global server_trainer_data = "INIT EMPTY"
global field_state = Dict{String, Any}() #team 1, team 2, ball
global state_lock = ReentrantLock()
global GLOBAL_TRAINER_SOCKET = UDPSocket()
#}}}
#{{{SOCCER CONSTANTS
const NUMBER_OF_PLAYERS = 6

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
	println("$port SENT \"$message\"")
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
	println("$port:\"$message\"")
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
					println("updated")
				end
			end
		end
	end
end
#{{{REDUNDANT
#=
function start_data_update_loop(sock::UDPSocket)
	Threads.@spawn begin
		while true
			send_command(TRAINER_PORT, sock, "(look)")
			#look_data_parse(get_data_primitive(sock))
			sleep(5)
			d = get_data_primitive(sock)
			look_data_parse(d)
			println("\t\tUPDATED: $(field_state)")
			#sleep(COMMAND_UPDATE_DELAY-UPDATE_DELAY_MARGIN)
		end
	end
end
function start_data_update_loop(sock::UDPSocket)
	global server_trainer_data[sock] = "x x x"
	Threads.@spawn begin
		while true
			send_command(TRAINER_PORT, sock, "(look)")
			look_data_parse(get_data_primitive(sock))
			println("\t\tUPDATED: $(server_trainer_data[sock])")
			#sleep(COMMAND_UPDATE_DELAY-UPDATE_DELAY_MARGIN)
		end
	end
end
function get_look_data_number(sock::UDPSocket)::UInt64
	println("get_look_data_number")
	try
		parse(UInt64, split(get_saved_server_trainer_data(sock)," ")[3])+1 #(ok look NUM ((goal...
	catch
		UInt64(0)
	end
end

function get_look_data(sock::UDPSocket)::String
	println("get_look_data")
	firsttime::Bool = false
	previous_look_number::UInt64 = 0
	previous_look_number = get_look_data_number(sock)
	if previous_look_number == 0
		firsttime = true
	end
	println("$(server_trainer_data)")
	println("SENDING###############################################")
	send_command_primitive(TRAINER_PORT, sock, "(look)")
	if firsttime
		while split(server_trainer_data*" a a a a"," ")[2] != "look" sleep(1);println("first") end #wait until "look" from server after non-look message from server
	else
		while previous_look_number == get_look_data_number(sock) sleep(1);println("not first") end #wait until the next "look" from the server
	end
	
	server_trainer_data
end
function update_look_data()
	println("get_newest_look_data")
	look_data_parse(get_look_data(GLOBAL_TRAINER_SOCKET))
end
=#
#}}}
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
		#println("parsing... $(raw[i])")

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
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.position.y  = parse(Float16, raw[i+4])
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.velocity.dx = parse(Float16, raw[i+6])
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.velocity.dy = parse(Float16, raw[i+7])
			global field_state[raw[i+1]].players[parse(UInt8, raw[i+2])].physical.angle       = parse(Float16, raw[i+5])
			i+=9
		else
			i+=1
		end
	end
	#println("parse returning")
end
#}}}
#}}}
#{{{TEAM INITIATION FUNCTIONS
function create_player(id::UInt8, name::String)::Player
	port=Client(PLAYER_PORT)
	send_command(PLAYER_PORT, port, "(init $name (version $VERSION))")
	info = Dict(); if id == 1 #the first player is always the goalie
		info["role"] = "goalie"
	end
	Player(
		id,
		port,
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
function PLAYER_goto(player::Player, position::Point, margin::Float16)
	Threads.@spawn begin
		#while abs(player.physical.position.x - position.x) > margin
		#	  abs(player.physical.position.y - position.y) > margin
		while true
			#println("READING ",field_state)
			lock(state_lock) do

				x = abs(player.physical.position.x - position.x)
				y = abs(player.physical.position.y - position.y)
				println(x, " ", y)
				
				println("goto loop")
				θ = rad2deg(atan((position.y + player.physical.position.y), (position.x - player.physical.position.x)))
				pangle = -player.physical.angle
				println(θ)
				println(pangle)
				Δθ = mod(pangle - θ + 180, 360) - 180
				println("Δθ = $Δθ")
				if abs(Δθ) > 5
					send_command(PLAYER_PORT, player.port, "(turn $(Δθ/2))")
					sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN)
				end
			end
			send_command(PLAYER_PORT, player.port, "(dash 100)")
			sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN)
			#=
			println(field_state)
			lock(state_lock) do

			x = abs(field_state[physical.position.x - position.x)
			y = abs(player.physical.position.y - position.y)
			println(x, y)
			
			println("goto loop")
			θ = rad2deg(atan(position.y - player.physical.position.y / position.x - player.physical.position.x))
			println(θ)
			println(player.physical.angle)
			println(abs(mod(player.physical.angle - θ + 180, 360) - 180))
			if abs(mod(player.physical.angle - θ + 180, 360) - 180) > 10
				send_command(PLAYER_PORT, player.port, "(turn $θ)")
			end
	end
			send_command(PLAYER_PORT, player.port, "(dash 100)")
			sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN)
			=#
		end
	end
end
#}}}
#{{{MASTER
function master()
	#define game
	#begin
		#define teams
		#begin
			teamnames = ("Team_A", "Team_B")
			trainer = Client(TRAINER_PORT)
			GLOBAL_TRAINER_SOCKET = trainer
			send_command(TRAINER_PORT, trainer, "(init $(teamnames[1]) (version $VERSION))")
			teams = (create_team(teamnames[1], :RIGHT),
					 create_team(teamnames[2], :LEFT))
			field_state[teamnames[1]] = teams[1]
			field_state[teamnames[2]] = teams[2]
	
			#define starting positions
			starting_positions = (
				Point(30,0), #goalie
				Point(15,20),
				Point(15,10),
				Point(15,0), ######### ADD/DEFINE "KICKER" POSITION
				Point(15,-10),
				Point(15,-20)
			)
		#end
	#end
	
	#initiate game
	#begin
		#initiate clients
		#begin
			#initiate trainer
			#trainer = Client(TRAINER_PORT)
			#GLOBAL_TRAINER_SOCKET = trainer
			#send_command(TRAINER_PORT, trainer, "(init $(teamnames[1]) (version $VERSION))")
		#end
		
		#move ball to the center of the field
		#send_command(TRAINER_PORT, trainer, "(move (ball) 0 5)")

		#move players into their starting positions
		#side = Int8(1)
		#for team ∈ teams
		#	for i = 1:NUMBER_OF_PLAYERS
		#		PLAYER_goto(team.players[i], Point(starting_positions[i].x*side, starting_positions[i].y), Float16(1))
		#	end
		#	side = -1
		#end
	#end

	#run game
	#println(field_state)
	#=sleep(1)
	send_command_primitive(TRAINER_PORT, trainer, "(look)")
	sleep(1)
	data=get_data_primitive(trainer)
	sleep(1)
	send_command_primitive(TRAINER_PORT, trainer, "(look)")
	sleep(1)
	data=get_data_primitive(trainer)
	sleep(1)
	send_command_primitive(TRAINER_PORT, trainer, "(look)")
	sleep(1)
	data=get_data_primitive(trainer)
	look_data_parse(data)
	println(field_state)
	=#
	start_data_update_loop(trainer)
	sleep(0)
	PLAYER_goto(teams[1].players[1], Point(0, 0), Float16(1.0))
	sleep(0)
	send_command(TRAINER_PORT, trainer, "(change_mode play_on)")
	#send_command(PLAYER_PORT,teams[1].players[1].port, "(turn 30)")
	sleep(20)

	#kill game
	for team ∈ teams
		for player ∈ team.players
    		die(player.port)
		end
	end
	die(trainer)
end
#}}}
#}}}
#{{{PROGRAM
master()
#}}}
