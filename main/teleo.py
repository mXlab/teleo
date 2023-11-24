import numpy as np
from enum import Enum
import signal
import argparse
from messaging import *

class AgentState(Enum):
  OPENED = 1
  CLOSED = 2

class Agent:
  def __init__(self, kit_id, stepsPerSecond = 5):
    self.stepsPerSecond = stepsPerSecond

    self.kit = MisBKit(kit_id)

    self.osc_helper = OscHelper("teleo-agent", "localhost", 8000, 8001)
    self.osc_helper.map("/pleasure", self.receive_pleasure)
    self.initialize()

  def initialize(self):
    self.trust = 0
    self.happiness = 0
    self.curiosity = 0
    self.state = AgentState.OPENED
    self.current_pleasure = 0
    self.kit.begin()

  def terminate(self):
    self.kit.terminate()

  def start(self):
    self.close() # start closed

  def step(self):
    self.kit.loop()
    
    self.debug()

    # Current instantaneous pleasure.
    pleasure = self.pleasure()

    # Pleasure makes happiness.
    self.addHappiness(pleasure * 0.1)

    # In the OPENED state the agent is relaxed, more pleasurable but also more vulnerable.
    if self.state == AgentState.OPENED:

      # If I am open and I get positive pleasure, I become more trusting.
      if pleasure > 0:
        self.addTrust(0.1)
      else:
        self.addTrust(-0.2)

      # This is too much, I am afraid: close!
      if self.isAfraid():
        self.close()

    # In the CLOSED state the agent is more tense, less pleasurable but also less vulnerable.
    else:
      # Slowly decrease happiness.
      self.addHappiness(-0.01)

      # # Slowly increase curiosity.
      # self.addCuriosity(np.random.uniform(0, 0.2))

      # If I trust more than my happiness, open up!
      # In other words: if my trust is greater than my happiness, I am willing to take the risk of opening up. However, if I am happy enough, it is not worth the risk.
      if (self.trust >= self.happiness):
        self.open()

    # Wait.
    time.sleep(1.0 / self.stepsPerSecond)

  # For now this returns a value between -1 and +1 representing the agent's instantaneous pleasure or pain.
  def pleasure(self):
    return self.current_pleasure
  
  def receive_pleasure(self, p):
    println("receive pleasure")
    self.current_pleasure = np.clip(p, -1, +1)
  
  def debug(self):
    print("AGENT =====================")
    print("trust: " + str(self.trust))
    print("happiness: " + str(self.happiness))
    print("curiosity: " + str(self.curiosity))
    print("state: " + str(self.state))
    print()          
  
  def propertyAdd(self, original, variationPerSecond):
    return np.clip(original + variationPerSecond / self.stepsPerSecond, 0, 1)
  
  def addTrust(self, variationPerSecond):
    self.trust = self.propertyAdd(self.trust, variationPerSecond)

  def addHappiness(self, variationPerSecond):
    self.happiness = self.propertyAdd(self.happiness, variationPerSecond)

  def addCuriosity(self, variationPerSecond):
    self.curiosity = self.propertyAdd(self.curiosity, variationPerSecond)
    
  def close(self):
    self.state = AgentState.CLOSED
    # self.curiosity = 0 # reset curiosity

  def open(self):
    self.state = AgentState.OPENED
    
  def isAfraid(self, threshold = 0.5):
    return self.trust <= threshold
  
  def isCurious(self, threshold = 0.5):
    return self.curiosity >= threshold
  
  def hasPleasure(self, threshold = 0.5):
    return self.happiness >= threshold


def interrupt(signup, frame):
    print("Exiting program...")
    agent.terminate()
    stop = True
    sys.exit()


if __name__ == '__main__':
    global agent
    signal.signal(signal.SIGINT, interrupt)

    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("kit_id", type=int, help="ID of the kit to run")
    parser.add_argument("--fps", type=int, help="Number of steps per second", default=5)

    # Parse arguments.
    args = parser.parse_args()

    agent = Agent(args.kit_id, stepsPerSecond = args.fps)

    # run_settings = yaml.load(open(args.run_file, 'r'), Loader=yaml.SafeLoader)
    # settings = yaml.load(open(args.settings_file, 'r'), Loader=yaml.SafeLoader)

    # # Create world.
    # world = World(settings)

    # # Create agents (for now just one agent).
    # behaviors = run_settings['behaviors']
    # robots = run_settings['robots']

    # manager = Manager(world, run_settings)
    # world.set_manager(manager)

    agent.start()

    while True:
        agent.step()


def main():

  agent = Agent()
  agent.initialize()
  agent.start()

  while True:
    agent.step()
