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
	θ::Float16
	v::Float16
end
struct Player
	id::UInt8
	position::Point
	velocity::Velocity
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
		sleep(COMMAND_UPDATE_DELAY) #wait so as to not send commands at the same time
		unlock(mutex_command_send[DataKey(port,sock)])
	end
end

function start_data_update_loop(sock::UDPSocket)
	global server_trainer_data[sock] = "x x x"
	Threads.@spawn begin
		while true	
			global server_trainer_data[sock] = get_data_primitive(sock)
			println("\t\tUPDATED: $(server_trainer_data[sock])")
			#sleep(COMMAND_UPDATE_DELAY-UPDATE_DELAY_MARGIN)
		end
	end
end

function get_saved_server_trainer_data(sock::UDPSocket)::String
	server_trainer_data[sock]
end

function get_look_data_number(sock::UDPSocket)::UInt64
	parse(UInt32,split(get_saved_server_trainer_data(sock)," ")[3]) #(ok look NUM ((goal...
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
function create_player(id::UInt8)::Player
	Player(
		id,
		Point(0, 0),
		Velocity(0, 0),
		Dict()
	)
end

function create_team(name::String)::Team
	Team(
		name,
		Vector([create_player(UInt8(i)) for i in 1:NUMBER_OF_PLAYERS])
	)
end
#}}}
#{{{MASTER
function master()
	#define game
	#begin
		#define teams
		#begin
			teamnames = ("Team_A", "Team_B")
			teams = Dict(
				teamnames[1] => create_team(teamnames[1]),
				teamnames[2] => create_team(teamnames[2])
			)

			#define players
			#begin
				players = ntuple(i->Client(PLAYER_PORT), NUMBER_OF_PLAYERS*2)
	
				#define goalies
				for name ∈ teamnames
						teams[name].players[1].info["role"] = "goalie"
				end
			#end
	
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
			#initiate trainers
			trainers = (Client(TRAINER_PORT), Client(TRAINER_PORT))
			
			start_data_update_loop(trainers[1])
			
			for (i, name) ∈ enumerate(teamnames)
				send_command(TRAINER_PORT, trainers[i], "(init $name (version $VERSION))")
			end
			
			#for name ∈ teamnames
			#	send_command(TRAINER_PORT, trainerB, "(init $name (version $VERSION))")
			#end
			
			#initiate players
			for name ∈ teamnames
				for id=1:NUMBER_OF_PLAYERS
					send_command(PLAYER_PORT, players[id], "(init $name (version $VERSION))")
				end
			end
		#end
		
		#move ball to the center of the field
		send_command(TRAINER_PORT, trainers[1], "(move (ball) 0 5)")

		#move players into their starting positions
		inv = Int8(1)
		for (i, name) ∈ enumerate(teamnames)
			println("\t\t\t",name)
			for id = 1:NUMBER_OF_PLAYERS
					send_command(TRAINER_PORT, trainers[i],
"(move (player $(teams[name].name) $(id)) $(inv*starting_positions[id].x) $(starting_positions[id].y))")
					println("DATA: ",get_look_data(trainers[1]))
			end
			inv = -1
		end
	#end

	#run game
	send_command(TRAINER_PORT, trainers[1], "(change_mode play_on)")
	sleep(10)

	#kill game
	for player ∈ players
    	die(player)
	end
	for trainer ∈ trainers
		die(trainer)
	end
end
#}}}
#}}}
#{{{PROGRAM
master()
#}}}
#}}}
