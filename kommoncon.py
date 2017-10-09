#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import zmq
from kommon import *

__author__  = ('Kaan Akşit')
__version__ = '0.1'

class server():
   def __init__(self,port):
       self.port    = port
       self.context = zmq.Context()
       self.socket  = self.context.socket(zmq.PUB)
       self.socket.bind("tcp://*:%s" % self.port)
   def send(self,topic,message):
       self.socket.send_string("%s %s" % (topic,message))
       return True
   def close(self):
       return self.socket.close()

class client():
   def __init__(self,ip,port,topics,timeout=10):
       self.ip      = ip
       self.port    = port
       self.timeout = timeout
       self.topics  = topics
       self.context = zmq.Context()
       self.socket  = self.context.socket(zmq.SUB)
       self.socket.connect("tcp://%s:%s" % (ip,port))
       self.subscribe(self.topics)
       self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
       self.socket.setsockopt(zmq.LINGER, 0)
   def subscribe(self,topics):
       self.socket.setsockopt_string(zmq.SUBSCRIBE, topics)
       return True
   def receive(self):
       self.string              = self.socket.recv(flags=zmq.NOBLOCK)
       self.topic, self.message = self.string.split()
       return self.topic, self.message
   def close(self):
       return self.socket.close()

# Elem terefiş, kem gözlere şiş!
if __name__ == '__main__':
    pass
