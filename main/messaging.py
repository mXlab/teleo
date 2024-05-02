import argparse
import time
import signal
import sys

from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
from osc4py3 import oscmethod as osm

from pythonosc import osc_message_builder, osc_bundle_builder
from pythonosc import udp_client

class OscHelper:
    def __init__(self, name, ip, send_port=8888, recv_port=8889):
        print("Creating OSC link at IP {} send = {} recv = {}".format(ip, send_port, recv_port))
        self.name = name
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.client = udp_client.SimpleUDPClient(ip, int(send_port))

        osc_udp_server("0.0.0.0", int(recv_port), self.server_name())

        self.maps = {}

        # Init OSC.
        osc_startup()

        # Send all paths to the dispatch() method with information.
        osc_method("*", self.dispatch, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_SRCIDENT + osm.OSCARG_DATA)

    def client_name(self):
        return self.name + "_client"

    def server_name(self):
        return self.name + "_server"

    def send_message(self, path, args):
        if not isinstance(args, list) and not isinstance(args, tuple):
            args = [ args ]
        # print("Sending message {} {} to {}".format(path, str(args), self.client_name()))
        self.client.send_message(path, args)
        
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
    def dispatch(self, address, ip, data):
        # If ip is tuple, take first part of tuple.
        if isinstance(ip, tuple):
            ip = ip[0]
        # Check if address matches and if IP corresponds: if so, call mapped function.
        if address in self.maps and (ip == self.ip or (ip == '127.0.0.1' and self.ip == 'localhost')):
            item = self.maps[address]
            func = item['function']
            if item['extra'] is None:
                func(data)
            else:
                func(data, item['extra'])
    
    def loop(self):
        osc_process()

class MisBKit:
    def __init__(self, id, **settings):
        # MisBKit id.
        self.id = id
        self.is_paired = None
        self.motor_ids = None
        
        # Create array of OscHelper objects for communicating with the robots.
        name = "mbk-" + str(id).zfill(2)
        ip = settings.get('ip', "192.168.0." + str(15 + id))
        send_port = settings.get('send_port', 8888)
        receive_port = settings.get('receive_port', 8889)
        self.osc_helper = OscHelper(name, ip, send_port, receive_port)

        # Map all the OSC addresses to the appropriate functions.
        self.osc_helper.map("/paired", self.receive_paired)
        self.osc_helper.map("/isPaired", self.receive_is_paired)
        self.osc_helper.map("/get/kit/ids", self.receive_motor_ids)

    def send(self, address, *args):
        print(args)
        self.osc_helper.send_message(address, args)

    def send_bundle(self, messages):
        self.osc_helper.send_bundle(messages)

    def dispatch(self, address, src_info, data):
        ip = src_info[0]
        self.osc_helper.dispatch(address, ip, data)

    def loop(self):
        self.osc_helper.loop()

    def begin(self):
        print("Begin, check is paired")
        if not self.is_paired:
            self.pair()
            time.sleep(0.5)
        self.isPaired()
        while self.is_paired is None:
            time.sleep(0.1)
            self.loop()
            print(".")
        # Get motors.
        print("Get motor ids")
        self.get_motor_ids()
        while self.motor_ids is None:
            time.sleep(0.1)
            self.loop()
        print("Motor ids")
        print(self.motor_ids)

    def terminate(self):
        osc_terminate()

    # Connection.

    def pair(self):
        self.send('/pair', self.id)

    def isPaired(self):
        self.send('/isPaired', self.id)

    # Configuration.

    # Information.
    def get_motor_ids(self):
        print("MOtor ids")
        self.send("/get/kit/ids")

    # def get_rssi(self):
    #     self.send("/get/rssi")

    # Control.

    def scan(self):
        self.send("/scan")

    def stop_all(self):
        self.send("/stop-all")

    def reboot(self):
        self.send("/reboot")

    # Motors.

    def wheel(self, motor_id, speed):
        self.send("/set/motor/wheel", motor_id, speed)

    def joint(self, motor_id, speed):
        self.send("/set/motor/joint", motor_id, speed)

    def speed(self, motor_id, speed):
        self.send("/set/motor/speed", motor_id, speed)

    def stop(self, motor_id):
        self.send("/set/motor/stop", motor_id)

    # def get_temperature(get_temperature):
    #     self.send("/get/temperature", get_temperature)

    # def test_joint(test_joint):
    #     self.send("/test/joint", test_joint)

    # def test_wheel(test_wheel):
    #     self.send("/test/wheel", test_wheel)

    # def is_paired(is_paired):
    #     self.send("/isPaired", is_paired)

    # def get_rssi(send_signal_strength):
    #     self.send("/get/rssi", send_signal_strength)

    # def get_ids(get_ids):
    #     self.send("/get/ids", get_ids)

    # def set_id(set_misb_kit_id):
    #     self.send("/set/id", set_misb_kit_id)

    # def set_kit_ip(set_misb_kit_ip):
    #     self.send("/set/kit/ip", set_misb_kit_ip)

    # def set_kit_port(set_misb_kit_port):
    #     self.send("/set/kit/port", set_misb_kit_port)

    # def set_sensor_a2(set_sensor_a2):
    #     self.send("/set/sensor/A2", set_sensor_a2)

    # def set_sensor_a3(set_sensor_a3):
    #     self.send("/set/sensor/A3", set_sensor_a3)

    # def set_sensor_a4(set_sensor_a4):
    #     self.send("/set/sensor/A4", set_sensor_a4)

    # def set_sensor_a7(set_sensor_a7):
    #     self.send("/set/sensor/A7", set_sensor_a7)

    # def set_sensor_a9(set_sensor_a9):
    #     self.send("/set/sensor/A9", set_sensor_a9)

    # def set_smooth_sensor_a2(set_smooth_sensor_a2):
    #     self.send("/set/sensor/A2/smooth", set_smooth_sensor_a2)

    # def set_smooth_sensor_a3(set_smooth_sensor_a3):
    #     self.send("/set/sensor/A3/smooth", set_smooth_sensor_a3)

    # def set_smooth_sensor_a4(set_smooth_sensor_a4):
    #     self.send("/set/sensor/A4/smooth", set_smooth_sensor_a4)

    # def set_smooth_sensor_a7(set_smooth_sensor_a7):
    #     self.send("/set/sensor/A7/smooth", set_smooth_sensor_a7)

    # def set_smooth_sensor_a9(set_smooth_sensor_a9):
    #     self.send("/set/sensor/A9/smooth", set_smooth_sensor_a9)

    # def set_sensor_accel(set_sensor_accel):
    #     self.send("/set/sensor/accel", set_sensor_accel)

    # def set_sensor_dist(set_sensor_dist):
    #     self.send("/set/sensor/dist", set_sensor_dist)

    # def set_smooth_sensor_accel(set_smooth_sensor_accel):
    #     self.send("/set/sensor/accel/smooth", set_smooth_sensor_accel)

    # def set_smooth_sensor_dist(set_smooth_sensor_dist):
    #     self.send("/set/sensor/dist/smooth", set_smooth_sensor_dist)

    # def set_sensor_d1(set_sensor_d1):
    #     self.send("/set/sensor/D1", set_sensor_d1)

    # def set_sensor_d2(set_sensor_d2):
    #     self.send("/set/sensor/D2", set_sensor_d2)

    # def set_sensor_d3(set_sensor_d3):
    #     self.send("/set/sensor/D3", set_sensor_d3)

    # def set_sensor_d4(set_sensor_d4):
    #     self.send("/set/sensor/D4", set_sensor_d4)

    # def set_sensor_rate(set_sensor_rate):
    #     self.send("/set/sensor/rate", set_sensor_rate)

    # def set_sensor_save(save_config):
    #     self.send("/set/sensor/save", save_config)

    def receive_paired(self, id):
        if self.id == id:
          self.is_paired = True

    def receive_is_paired(self, is_paired):
        self.is_paired = is_paired

    def receive_motor_ids(self, ids):
        print("GDSFDSDF")
        print(ids)
        self.motor_ids = ids

def interrupt(signup, frame):
    global my_world, stop
    print("Exiting program...")
    my_world.terminate()
    stop = True
    sys.exit()


if __name__ == '__main__':
    import yaml
    import world

    signal.signal(signal.SIGINT, interrupt)

    stop = False
    settings = yaml.load(open('settings.yml', 'r'), Loader=yaml.SafeLoader)
    my_world = world.World(settings)
    while not stop:
        my_world.step()
        my_world.debug()
#        time.sleep(0.5)

