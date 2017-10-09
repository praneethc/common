#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,socket,time,_thread
from kommon import *
import kommoncon

__author__  = ('Kaan Akşit')
__version__ = '0.1'


# Main definition,
def main(port="1234",title="CHAT"):
    nick = input("Nickname: ")
    s    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("255.255.255.254",80))
    ip   = s.getsockname()[0]
    s.close()
    prompt("A server will start on %s:%s" % (ip,port),title=title)
    server = kommoncon.server(port)
    prompt("Connected to  %s:%s" % (sys.argv[1],sys.argv[2]),title=title)
    client = kommoncon.client(sys.argv[1],sys.argv[2],"all")
    prompt("Server started!",title=title)
    prompt("Press s key to send message.",title=title)
    prompt("Press q key to exit the application.", title=title)
    _thread.start_new_thread(read,(client,title,))
    while True:
       key = getch()
       if (key == 's'):
           message  = input("Type message: ")
           send_msg = '%s -- %s' % (nick,message)
           server.send("all",send_msg)
           prompt("Message sent: %s" % message, title=title)
       elif (key == 'q'):
           sys.exit()
           server.close()
           client.close()
    return True

def read(client,title):
    while True:
        topic,msg = client.receive()
        prompt("Message received : %s" %msg, title=title)


# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    sys.exit(main())

