using Sockets
using Dates

global mutex_command_send = IdDict{UDPSocket, ReentrantLock}()

global socks_last_sent = Dict() #last time each socket (player) sent a command
global COMMAND_UPDATE_DELAY = 0.11 #command-sending delay (seconds)

const PORT=6001

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
end

function get_sensordata(sock::UDPSocket)
	return String(recv(sock))
end

function die(sock::UDPSocket)
    close(sock)
end

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

