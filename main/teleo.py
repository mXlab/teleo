import numpy as np
from enum import IntEnum
import signal
import argparse
from messaging import *

class AgentState(IntEnum):
  CLOSED   = 0
  OPENED   = 1  
  N_STATES = 2

  def switch(state):
    return AgentState.OPENED if state == AgentState.CLOSED else AgentState.CLOSED

class AgentAction(IntEnum):
  STAY      = 0
  CHANGE    = 1
  N_ACTIONS = 2

class Agent:
  def __init__(self, kitId = None, stepsPerSecond = 5):
    self.stepsPerSecond = stepsPerSecond

    if kitId is None:
      self.kit = None
    else:
      self.kit = MisBKit(kitId)

    self.oscHelper = OscHelper("teleo-agent", "localhost", send_port=8000, recv_port=8001)
    self.oscHelper.map("/pleasure", self.receivePleasure)
    # self.oscHelper.map("/trust", self.receiveTrust)

    self.learningRate = 0.1

    self.rewardWeightState  = 0.3
    self.rewardWeightAction = 0.1
    self.rewardWeightTrust  = 0.6

    self.initialize()

  # Initialize the agent.
  def initialize(self):
    # Init properties.
    self.happiness = 0
    self.curiosity = 0
    self.trust     = 0
    # Init priors.
    self.stateValues = np.full(AgentState.N_STATES, 0)
    # Starting state.
    self.state = AgentState.CLOSED
    self.currentPleasure = 0 # latest registered pleasure
    # Start comm with MisBKit.
    if self.kit is not None:
      self.kit.begin()

  def terminate(self):
    if self.kit is not None:
      self.kit.terminate()

  def start(self):
    self.close() # start closed

  def step(self):
    self.debug()

    # Current instantaneous pleasure.
    pleasure = self.pleasure()
    self.resetPleasure()

    # Pleasure makes happiness.
    reward = pleasure

    # In the OPENED state the agent is relaxed, more pleasurable but also more vulnerable.
    if self.state == AgentState.OPENED:
      
      # If I am open and I get positive pleasure, I become more trusting.
      if pleasure > 0:
        self.addTrust(0.1)
      elif pleasure < 0:
        self.addTrust(-0.2)

      # self.learn(self.trust)

      # # This is too much, I am afraid: close!
      # if self.isAfraid():
      #   self.close()

    # In the CLOSED state the agent is more tense, less pleasurable but also less vulnerable.
    else:
      # Slowly decrease happiness (energy consumption).
      reward -= 0.1

      # # Slowly increase curiosity.
      # self.addCuriosity(np.random.uniform(0, 0.2))

      # # If I trust more than my happiness, open up!
      # # In other words: if my trust is greater than my happiness, I am willing to take the risk of opening up. However, if I am happy enough, it is not worth the risk.
      # if (self.objective_trust + self.currentHumanTrust() >= self.happiness):
      #   self.open()

    # Update.
    self.addHappiness(reward)
    self.updateStateValue(reward)

    # Take decision.
    action = self.chooseAction(self.state)
    
    # Act.
    # TODO: implement action

    # Switch state.
    self.state = AgentState.switch(self.state) if action == AgentAction.CHANGE else self.state
    if action == AgentAction.CHANGE:
      self.curiosity = 0
    else:
      self.addCuriosity(0.1)

    # Wait.
    startTime = time.time()
    while time.time() - startTime < (1.0 / self.stepsPerSecond):
      if self.kit is not None:
        self.kit.loop()
      self.oscHelper.loop()

  # For now this returns a value between -1 and +1 representing the agent's instantaneous pleasure or pain.
  def pleasure(self):
    return self.currentPleasure if self.currentPleasure is not None else 0
  
  # def trust(self):
  #   return self.trust
  
  def resetPleasure(self):
    self.currentPleasure = None

  def updateStateValue(self, value, alpha = 0.1):
    # Update state values using moving average.
    self.stateValues[self.state.value] -= alpha * (self.stateValues[self.state.value] - value)

  def receivePleasure(self, data):
    p = data[0]
    self.currentPleasure = np.clip(p, -1, +1)

  def chooseAction(self, state):
    # Evaluate options.
    values = []
    for action in range(AgentAction.N_ACTIONS):
      values.append(self.evaluate(state, action))

    self.oscHelper.send_message("/action-values", values)

    values = np.array(values)
    print("Values: " + str(values)) 

    # Choose action.
    return np.argmax(values)

  def evaluate(self, state, action):
    # Init value of state, action.
    value = 0

    # Compute useful values.
    nextState = AgentState.switch(state) if action == AgentAction.CHANGE else state

    # Contribution of prior of next state.
    value += self.rewardWeightState * self.stateValues[nextState]

    # Contribution of action.

    # The CHANGE action is more likely to be taken if I am curious.
    if action == AgentAction.CHANGE:
      if nextState == AgentState.OPENED:
        actionReward = self.curiosity
      else:  # curiosity drives me only parially if I am going to be CLOSED
        actionReward = 0.6 * self.curiosity
    # The STAY action is NOT driven by curiosity but by preserving energy.
    else:
      actionReward = 0.5 # costs less energy
    value += self.rewardWeightAction * actionReward

    # Contribution of trust: actions leading to OPENED are more likely to be taken if I am trustful.
    trustContribution = self.rewardWeightTrust * self.trust 
    # Trust contribution is inverted if next state is CLOSED.
    if nextState == AgentState.CLOSED:
      trustContribution *= -1

    value += trustContribution

    # Return value.
    return value

  def sendState(self):
    self.oscHelper.send_bundle({
      "/trust": self.trust,
      "/happiness": self.happiness,
      "/curiosity": self.curiosity,
      "/state": self.state
    })

  def debug(self):
    print("AGENT =====================")
    print("trust: " + str(self.trust))
    print("happiness: " + str(self.happiness))
    print("curiosity: " + str(self.curiosity))
    print("state: " + str(self.state))
    print("state values: " + str(self.stateValues))
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
    parser.add_argument("--kit-id", type=int, help="ID of the kit to run", default=0)
    parser.add_argument("--simulation-mode", type=bool, help="Simulation mode (no MisBKit)", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--fps", type=int, help="Number of steps per second", default=5)

    # Parse arguments.
    args = parser.parse_args()

    kitId = args.kit_id if not args.simulation_mode else None
    agent = Agent(kitId, stepsPerSecond = args.fps)

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
        agent.sendState()

