#{{{MODULES
using Sockets
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
global mutex_command_send = IdDict{UDPSocket, ReentrantLock}()
global server_trainer_data = IdDict{UDPSocket, String}()
#}}}

#{{{SOCCER CONSTANTS
const NUMBER_OF_PLAYERS = 6
#}}}

#{{{STRUCTURES
struct Coordinate
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
	teamname::String
	players::Vector{Player}
end
#}}}

#{{{UDP PRIMITIVE
function Client(server::String="localhost", port::UInt16)::UDPSocket #initiate a client/socket (player)
    sock = UDPSocket()
    bind(sock, IP, 0)
    println("Connected to $server:$port")
	socks_last_sent[sock] = Float64(0) #add player to dictionary
	mutex_command_send[sock] = ReentrantLock()
    return sock
end

function send_command_primitive(port::UInt16, sock::UDPSocket, message::String)
	send(sock, IP, port, message)
end

function get_data_primitive(sock::UDPSocket)::String
	String(recv(sock))
end

function die(sock::UDPSocket)
    close(sock)
end
#}}}

#{{{CUSTOM COMMUNICATION PROTOCOLS
function send_command(port::UInt16,sock::UDPSocket, message::String)
	Threads.@spawn begin
		lock(mutex_command_send[sock])
		send_command_primitive(port,sock,message)
		sleep(COMMAND_UPDATE_DELAY) #wait so as to not send commands at the same time
		unlock(mutex_command_send[sock])
	end
end

function start_data_update_loop(sock::UDPSocket)
	Threads.@spawn begin
		while true	
			global server_trainer_data[sock] = get_data_primitive(sock) #POTENTIAL BUG
			println("\t\tUPDATED: $(server_trainer_data[sock])")
			#sleep(COMMAND_UPDATE_DELAY-UPDATE_DELAY_MARGIN)
		end
	end
end

function get_saved_server_trainer_data(sock::UDPSocket)::String
	server_trainer_data[sock]
end

function get_look_data_number(sock::UDPSocket)::UInt64
	parse(UInt32,split(get_data(sock)," ")[3]) #(ok look NUM ((goal...
end

function get_look_data(sock::UDPSocket)::String
	firsttime::Bool = false
	previous_look_number::UInt32 = 0
	try
		previous_look_number = look_number(sock)
	catch
		firsttime = true
	end
	send_command(sock, "(look)")
	if firsttime
		while split(get_saved_server_trainer_data(sock)," ")[2] != "look" sleep(0.01) end #wait until "look" from server after non-look message from server
	else
		while previous_look_number == get_look_data_number(sock) sleep(0.01) end #wait until the next "look" from the server
	end
	
	get_saved_server_trainer_data(sock)
end
#}}}

function create_player(id::UInt8)::Player
	Player(
		id,
		Point(0, 0),
		Velocity(0, 0),
		Dict()
	)
end

function create_team(teamname::String)::Team
	Team(
		teamname,
		Vector([create_player(Int8(i)) for i in 1:NUMBER_OF_PLAYERS])
	)
end



function master()
	#initiate trainer
	trainer = Client()
	start_data_update_loop(trainer)
	send_command(TRAINER_PORT, trainer, "(init Team_A (version $VERSION))")
	
	#initiate players
	players = ntuple(i->Client(), NUMBER_OF_PLAYERS*2)
	for player ∈ players
		send_command(PLAYER_PORT, player, "(init Team_A (version $VERSION))")
	end

	#define teams
	begin
		teamnames = ("Team_A", "Team_B")
		teams = Dict(
			teamnames[1] => create_team(teamnames[1]),
			teamnames[2] => create_team(teamnames[2])
		)
	
		#define goalies
		for team ∈ teams
			team.players[1].info["role"] = "goalie"
		end
	end
	
	#move ball to the center of the field
	send_command(trainer, "(move (ball) 0 0)")
	
	#move players into position
	for team ∈ teams
		send_command(trainer, "(move (player $team 1) 10 0)")
	end
	
	for i=1:20
		send_command(trainer, "(look)")
		println(get_sensordata(trainer))
		sleep(1)
	end

   	die(trainer)

	sleep(3)
	#send_command(trainer, "(move (GameObject (FieldObject (MobileObject (Ball)))) 10 10)")
	send_command(trainer, "(move (ball) 0 0)")
	send_command(trainer, "(move (player Team_A 1) 10 0)")
	send_command(trainer, "(move (player Team_A 2) 10 10)")
	send_command(trainer, "(move (player Team_A 3) 10 20)")
	send_command(trainer, "(move (player Team_A 4) 10 30)")
	send_command(trainer, "(move (player Team_A 5) 10 -10)")
	send_command(trainer, "(move (player Team_A 6) 10 -20)")
	send_command(trainer, "(change_mode play_on)")

	for i=1:15
		println("LOOK: ",look(trainer))
		sleep(1.5)
	end
	
   	die(trainer)
end

master()


function master()
	
	sleep(3)
	for i in 1:100
		for player ∈ players
			send_command(player, "(dash 100)")
			send_command(player, "(turn 10)")
		end
    end
	
	sleep(10)
	for player ∈ players
    	die(player)
	end
end

master()


global socks_last_sent = Dict() #last time each socket (player) sent a command
global COMMAND_UPDATE_DELAY = 0.11 #command-sending delay (seconds)




function send_command(sock::UDPSocket, message::String)
	Threads.@spawn begin
		lock(mutex_command_send[sock])
		send_command_primitive(sock,message)
		sleep(COMMAND_UPDATE_DELAY) #wait so as to not send commands at the same time
		unlock(mutex_command_send[sock])
	end
end




function main()
	trainer = Client()

	send_command(trainer, "(init A (verion 15))")
	sleep(1)
	send_command(trainer, "(change_mode play_on)")

	for i=1:20
		send_command(trainer, "(look)")
		println(get_sensordata(trainer))
		sleep(1)
	end

   	die(trainer)
end

main()

