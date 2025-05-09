from osc4py3.as_eventloop import *
# from osc4py3 import oscbuildparse
from osc4py3 import oscmethod as osm

from pythonosc import osc_message_builder, osc_bundle_builder
from pythonosc import udp_client

import socket

class BroadcastOSCClient:
    def __init__(self, ip: str, port: int):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send_message(self, address: str, args):
        builder = osc_message_builder.OscMessageBuilder(address=address)
        if not isinstance(args, list):
            args = [args]
        for arg in args:
            builder.add_arg(arg)
        msg = builder.build()
        self.sock.sendto(msg.dgram, self.addr)

    # Sends an OSC message or bundle over the socket.
    # Accepts an OscMessage or OscBundle (from python-osc).
    def send(self, osc_obj):
        try:
            dgram = osc_obj.dgram  # This works for both messages and bundles
        except AttributeError:
            raise TypeError("send() requires an OSC object with a .dgram attribute")
        
        self.sock.sendto(dgram, self.addr)

class OscHelper:
    def __init__(self, name, ip, send_port, recv_port, redirect_ip=None, redirect_port=8001):
        print("Creating OSC link at IP {} send = {} recv = {}".format(ip, send_port, recv_port))
        self.name = name
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.client = BroadcastOSCClient(ip, int(send_port))
        if redirect_ip is not None:
            self.client_redirect = BroadcastOSCClient(redirect_ip, int(redirect_port))
        else:
            self.client_redirect = None
        osc_udp_server("127.0.0.1", int(recv_port), self.server_name())

        osc_method("*", self.dispatch, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_SRCIDENT + osm.OSCARG_DATAUNPACK)
        self.maps = {}

    def client_name(self):
        return self.name + "_client"

    def server_name(self):
        return self.name + "_server"

    def send_message(self, path, args):
        if not isinstance(args, list):
            args = [ args ]
        self.client.send_message(path, args)
        # print("Sending message {} {} to {}".format(path, str(args), self.client_name()))

    # Send group of messages as a bundle.
    def send_bundle(self, messages):
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        for path, args in messages.items():
            print("Sending bundle with {}".format(path))
            if not isinstance(args, list):
                args = [ args ]
            msg_builder = osc_message_builder.OscMessageBuilder(address=path)
            for a in args:
                msg_builder.add_arg(a)
            bundle.add_content(msg_builder.build())
        self.client.send(bundle.build())

    # Adds an OSC path by assigning it to a function, with optional extra data.
    def map(self, path, function, extra=None):
        self.maps[path] = { 'function': function, 'extra': extra }

    # Dispatches OSC message to appropriate function, if it corresponds to helper.
    def dispatch(self, address, srcident, data):
        print("Received {} from {}".format(address, srcident))
        # Redirect
        if self.client_redirect is not None:
            self.client_redirect.send_message(address, data)
        # Check if address matches and if IP corresponds: if so, call mapped function.
        ip, __ = srcident
        if address in self.maps and (ip == self.ip or (ip == '127.0.0.1' and self.ip == 'localhost')):
            item = self.maps[address]
            func = item['function']
            if item['extra'] is None:
                func(data)
            else:
                func(data, item['extra'])
