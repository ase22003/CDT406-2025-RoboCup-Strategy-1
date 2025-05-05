using Sockets
using Dates

global mutex_command_send = IdDict{UDPSocket, ReentrantLock}()

global socks_last_sent = Dict() #last time each socket (player) sent a command
global COMMAND_UPDATE_DELAY = 0.11 #command-sending delay (seconds)

function Client(server::String="localhost", port::Int=6000) #initiate a client/socket (player)
    sock = UDPSocket()
    bind(sock, ip"127.0.0.1", 0)
    println("Connected to $server:$port")
	socks_last_sent[sock] = Float64(0) #add player to dictionary
	mutex_command_send[sock] = ReentrantLock()
    return sock
end

function send_command_primitive(sock::UDPSocket, message::String)
	send(sock, ip"127.0.0.1", 6000, message)
end

function get_sensordata(sock::UDPSocket)
	return String(recv(sock))
end

function die(sock::UDPSocket)
    close(sock)
end

#{{{deprecated
function OLD_send_command(sock::UDPSocket, message::String)
	#wait until it has been at least COMMAND_UPDATE_DELAY seconds since last send-off
	last_run_time = socks_last_sent[sock]
	println(last_run_time)
	current_time = time()
	if current_time - last_run_time < COMMAND_UPDATE_DELAY
        sleep(COMMAND_UPDATE_DELAY - (current_time - last_run_time))
    end
	socks_last_sent[sock] = time() #update time

	send_command_primitive(sock,message)
end
#}}}

function send_command(sock::UDPSocket, message::String)
	Threads.@spawn begin
		lock(mutex_command_send[sock])
		send_command_primitive(sock,message)
		sleep(COMMAND_UPDATE_DELAY) #wait so as to not send commands at the same time
		unlock(mutex_command_send[sock])
	end
end




function main()
	players = ntuple(i->Client(), 6)
	for player ∈ players
		send_command(player, "(init Team_A)")
		send_command(player, "(move 0 0)")
	end
	
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

main()
