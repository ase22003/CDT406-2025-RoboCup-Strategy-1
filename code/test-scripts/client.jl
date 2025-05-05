using Sockets
using Dates

function Client(server::String="localhost", port::Int=6000)
    sock = UDPSocket()
    bind(sock, ip"127.0.0.1", 0)
    println("Connected to $server:$port")
    return sock
end

function send_command(sock::UDPSocket, message::String)
    send(sock, ip"127.0.0.1", 6000, message)
end

function get_sensordata(sock::UDPSocket)
	return String(recv(sock))
end

function die(sock::UDPSocket)
    close(sock)
end

function main()
    sock1 = Client("localhost", 6000)
    sock2 = Client("localhost", 6000)
    sock3 = Client("localhost", 6000)
    sock4 = Client("localhost", 6000)
	send_command(sock1, "(init T1)")
	send_command(sock2, "(init T1)")
	send_command(sock3, "(init T2)")
	send_command(sock4, "(init T2)")
	sleep(1)	
	send_command(sock1, "(move 0 0)")
	send_command(sock2, "(move 0 0)")
	send_command(sock3, "(move 0 0)")
	send_command(sock4, "(move 0 0)")
		sleep(3)
    for i in 1:100
        sleep(0.1)
        #data = get_sensordata(sock)
        #println(data)
        send_command(sock1, "(turn 5)")
        send_command(sock2, "(turn 10)")
        send_command(sock3, "(turn -5)")
        send_command(sock4, "(turn -10)")
		sleep(0.1)
		send_command(sock1, "(dash 100)")
		send_command(sock2, "(dash 100)")
		send_command(sock3, "(dash 100)")
		send_command(sock4, "(dash 100)")
    end

    die(sock1)
    die(sock2)
    die(sock3)
    die(sock4)
end

main()

