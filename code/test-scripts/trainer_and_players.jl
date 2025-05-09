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
struct Point
	x::Float16
	y::Float16
end
struct Velocity
	dx::Float16
	dy::Float16
end
struct Physical
	position::Point
	velocity::Velocity
	angle::Float16
end
struct Player
	id::UInt8
	port::UDPSocket
	physical::Physical
	info::Dict{String, Any} #roles are given by "role" herein
end
struct Team
	name::String
	players::Vector{Player}
end
#}}}
#}}}
#{{{UDP CONSTANTS
const IP = ip"127.0.0.1"
const PLAYER_PORT  = 6000
const TRAINER_PORT = 6001
const VERSION=7
#}}}
#{{{SERVER COMMUNICATION DELAYS
const COMMAND_UPDATE_DELAY = 0.1 #command-sending delay (seconds)
const UPDATE_DELAY_MARGIN = 0.0
#}}}
#{{{GLOBALS
global mutex_command_send = IdDict{DataKey, ReentrantLock}()
global server_trainer_data = IdDict{UDPSocket, String}()
global field_state = Dict{String, Any}() #team 1, team 2, ball
#}}}
#{{{SOCCER CONSTANTS
const NUMBER_OF_PLAYERS = 6
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
	println("$port SENT \"$message\"!")
end

function get_data_primitive(sock::UDPSocket)::String
	String(recv(sock))
end

function die(sock::UDPSocket)
    close(sock)
end
#}}}
#{{{CUSTOM COMMUNICATION PROTOCOLS
function send_command(port::Int,sock::UDPSocket, message::String)
	println("$port:\"$message\"")
	Threads.@spawn begin
		lock(mutex_command_send[DataKey(port,sock)])
		send_command_primitive(port,sock,message)
		sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN) #wait so as to not send commands at the same time
		unlock(mutex_command_send[DataKey(port,sock)])
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

function look_data_parse(raw::String, teams::Tuple{Team, Team})
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
							   parse(Float16, raw[i]),
						 	   parse(Float16, raw[i+1])),
						  Velocity(
								   parse(Float16, raw[i+2]),
						  		   parse(Float16, raw[i+3])))
			i+=4
		else if raw[i] == "player"
			global field_state[raw[i+1]].players[raw[i+2]].physical.position.x  = parse(Float16, raw[i+3])
			global field_state[raw[i+1]].players[raw[i+2]].physical.position.y  = parse(Float16, raw[i+4])
			global field_state[raw[i+1]].players[raw[i+2]].physical.velocity.dx = parse(Float16, raw[i+6])
			global field_state[raw[i+1]].players[raw[i+2]].physical.velocity.dy = parse(Float16, raw[i+5])
			global field_state[raw[i+1]].players[raw[i+2]].physical.angle       = parse(Float16, raw[i+7])
			i+=9
		else
			i+=1
		end
	end
end

function get_saved_server_trainer_data(sock::UDPSocket)::String
	server_trainer_data[sock] * " a a a a"
end

function get_look_data_number(sock::UDPSocket)::UInt64
	try
		parse(UInt64, split(get_saved_server_trainer_data(sock)," ")[3])+1 #(ok look NUM ((goal...
	catch
		UInt64(0)
	end
end

function get_look_data(sock::UDPSocket)::String
	firsttime::Bool = false
	previous_look_number::UInt64 = 0
	try
		previous_look_number = get_look_data_number(sock)
	catch
		firsttime = true
	end
	send_command(TRAINER_PORT, sock, "(look)")
	if firsttime
		while split(get_saved_server_trainer_data(sock)," ")[2] != "look" sleep(0.01) end #wait until "look" from server after non-look message from server
	else
		while previous_look_number == get_look_data_number(sock) sleep(0.01) end #wait until the next "look" from the server
	end
	
	get_saved_server_trainer_data(sock)
end
#}}}
#{{{TEAM INITIATION FUNCTIONS
function create_player(id::UInt8, name::String)::Player
	port=Client(PLAYER_PORT)
	send_command(PLAYER_PORT, port, "(init $name (version $VERSION))")
	info = Dict() if id == 1 #the first player is always the goalie
		info["role"] = "goalie"
	end
	Player(
		id,
		port,
		Physical(Point(0, 0), Velocity(0, 0)),
		info
	)
end

function create_team(name::String)::Team
	Team(
		name,
		Vector([create_player(UInt8(i), name) for i in 1:NUMBER_OF_PLAYERS])
	)
end
#}}}
#{{{LOW LEVEL SKILLS
function PLAYER_goto(player::Player, position::Point)
	Thread.@spawn begin
		while
	end
#}}}
#{{{MASTER
function master()
	#define game
	#begin
		#define teams
		#begin
			teamnames = ("Team_A", "Team_B")
			teams = (create_team(teamnames[1]),
					 create_team(teamnames[2]))
	
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
			trainer = Client(TRAINER_PORT)
			
			start_data_update_loop(trainer)
			send_command(TRAINER_PORT, trainer, "(init $(teamnames[1]) (version $VERSION))")
		#end
		
		#move ball to the center of the field
		#send_command(TRAINER_PORT, trainer, "(move (ball) 0 5)")

		#move players into their starting positions
		side = Int8(1)
		for team ∈ teams
			for i = 1:NUMBER_OF_PLAYERS
				PLAYER_goto(team.players[i], Point(starting_positions[i].x*side, starting_positions[i].y))
			end
			side = -1
		end
	#end

	#run game
	send_command(TRAINER_PORT, trainer, "(change_mode play_on)")
	sleep(10)

	#kill game
	for player ∈ team.players
    	die(player.port)
	end
	die(trainer)
end
#}}}
#}}}
#{{{PROGRAM
master()
#}}}
