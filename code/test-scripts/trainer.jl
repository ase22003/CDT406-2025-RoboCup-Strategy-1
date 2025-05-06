using Sockets
using Dates

global mutex_command_send = IdDict{UDPSocket, ReentrantLock}()

global socks_last_sent = Dict() #last time each socket (player) sent a command
const COMMAND_UPDATE_DELAY = 0.1 #command-sending delay (seconds)
const UPDATE_DELAY_MARGIN = 0.0

const PORT=6001
const VERSION=7

global server_data = IdDict{UDPSocket, String}()

function Client(server::String="localhost", port::Int=PORT) #initiate a client/socket (player)
    sock = UDPSocket()
    bind(sock, ip"127.0.0.1", 0)
    println("Connected to $server:$port")
	socks_last_sent[sock] = Float64(0) #add player to dictionary
	mutex_command_send[sock] = ReentrantLock()
    return sock
end

function send_command_primitive(sock::UDPSocket, message::String)
	send(sock, ip"127.0.0.1", PORT, message)
	println("\tSENT \"",message,'\"')
end

function get_data_primitive(sock::UDPSocket)
	return String(recv(sock))
end

function die(sock::UDPSocket)
    close(sock)
end

function send_command(sock::UDPSocket, message::String)
	Threads.@spawn begin
		lock(mutex_command_send[sock])
		send_command_primitive(sock,message)
		sleep(COMMAND_UPDATE_DELAY+UPDATE_DELAY_MARGIN) #wait so as to not send commands at the same time
		unlock(mutex_command_send[sock])
	end
end

function start_data_update_loop(sock::UDPSocket)
	Threads.@spawn begin
		while true	
			global server_data[sock] = get_data_primitive(sock)
			println("\t\tUPDATED: $(server_data[sock])")
			#sleep(COMMAND_UPDATE_DELAY-UPDATE_DELAY_MARGIN)
		end
	end
end

function get_data(sock::UDPSocket)
	return server_data[sock]
end

function look_number(sock::UDPSocket)
	return parse(UInt32,split(get_data(sock)," ")[3]) #(ok look NUM ((goal...
end

function look(sock::UDPSocket)
	firsttime::Bool = false
	previous_look_number::UInt32 = 0
	try
		previous_look_number = look_number(sock)
	catch
		firsttime = true
	end
	send_command(sock, "(look)")
	if firsttime
		while split(get_data(sock)," ")[2] != "look" sleep(0.01) end #wait until "look" from server after non-look message from server
	else
		while previous_look_number == look_number(sock) sleep(0.01) end #wait until the next "look" from the server
	end
	return get_data(sock)
end




function master()
	trainer = Client()
	start_data_update_loop(trainer)
	
	send_command(trainer, "(init Team_A (version $VERSION))")
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
