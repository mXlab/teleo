"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF


class MotorExt:
	"""
	MotorExt description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.osc = parent.Mbk.op('COM').op('oscout1')
		self.lastSpeed = 0
		# properties
		TDF.createProperty(self, 'MyProperty', value=0, dependable=True,
						   readOnly=False)

		# attributes:
		self.a = 0 # attribute
		self.B = 1 # promoted attribute
		self.id = self.ownerComp.par.Id.val
		#self.interface = parent.Project.par.Interface.val
		

		# stored items (persistent across saves and re-initialization):
		storedItems = [
			# Only 'name' is required...
			{'name': 'StoredProperty', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
		]
		# Uncomment the line below to store StoredProperty. To clear stored
		# 	items, use the Storage section of the Component Editor
		
		# self.stored = StorageManager(self, ownerComp, storedItems)

		#self.serial = parent.Project.op('Serial_interface').op('serial1')

	def isInterfaceSerial(self):
			# if self.interface == "Serial":
			# 	return True
			# else:
			return False

## DIRECT CONTROL
	def Stop(self):
		# if self.isInterfaceSerial():
			
		# 	cmd ="104" + str(self.id).zfill(3)+"0"
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/set/motor/stop',[self.id])
		return



	def Wheel(self,speed):
		#debug("Changing motor {} wheel speed to {}".format(self.id,speed))

		# if self.isInterfaceSerial():
		# 	if speed < 0 :
		# 		speed = -speed * 1023
		# 	elif speed > 0 :
		# 		speed = 1024 + speed * 1023
		# 	cmd ="104" + str(self.id).zfill(3)+str(int(speed))
		# 	self.serial.send(cmd)
		# else:
		
		self.osc.sendOSC('/set/motor/wheel',[self.id,float(speed)])
		return
	
	def Speed(self,speed):
		#debug("Changing motor {} speed to {}".format(self.id,speed))
		# if self.isInterfaceSerial():
		# 	cmd ="107" + str(self.id).zfill(3)+str(int(tdu.remap(speed,0.0,1.0,0.0,1023.0)))
		# 	self.serial.send(cmd)
		# else:
		speed = float(speed)
		if speed == self.lastSpeed:
			return
		else:
			self.osc.sendOSC('/set/motor/speed',[self.id,speed])
			self.lastSpeed = speed
		return

	def Joint(self,goal):
		
		
		#debug("Changing motor {} goal position to {} ".format(self.id,goal,))
		# if self.isInterfaceSerial():
		# 	cmd ="105" + str(self.id).zfill(3)+str(int(tdu.remap(goal,-1.0,1.0,0.0,1023.0)))
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/set/motor/joint',[self.id,float(goal)])
		return

##TESTS

	def TestWheel(self):
		#debug("Asking motor {} to test wheel mode".format(self.id))
		# if self.isInterfaceSerial():
		# 	cmd ="201" + str(self.id).zfill(3)
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/test/wheel',[self.id])
		return

	def TestJoint(self):
		#debug("Asking motor {} to test joint mode".format(self.id))
		# if self.isInterfaceSerial():
		# 	cmd ="200" + str(self.id).zfill(3)
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/test/joint',[self.id])
		return
	
##GETTERS

	def GetModel(self):
		#debug("Asking motor {} for model number".format(self.id))
		# if self.isInterfaceSerial():
		# 	cmd ="301" + str(self.id).zfill(3)
		# 	print(cmd)
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/get/motor/model',[self.id])
		return
	
	def GetFirmware(self):
		#debug("Asking motor {} for firmware version".format(self.id))
		# if self.isInterfaceSerial():
		# 	cmd ="302" + str(self.id).zfill(3)
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/get/motor/firmware',[self.id])
		return
	
	def GetTemperature(self):
		#debug("Asking motor {} for current temperature".format(self.id))
		if self.isInterfaceSerial():
			cmd ="306" + str(self.id).zfill(3)
			self.serial.send(cmd)
		else:	
			self.osc.sendOSC('/get/motor/temperature',[self.id])
		return

	def GetBaudrate(self):
		#debug("Asking motor {} for baudrate".format(self.id))
		# if self.isInterfaceSerial():
		# 	cmd ="304" + str(self.id).zfill(3)
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/get/motor/baudrate',[self.id])
		return
	
	def GetDelaytime(self):
		#debug("Asking motor {} for delay time".format(self.id))
		# if self.isInterfaceSerial():
		# 	cmd ="305" + str(self.id).zfill(3)
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/get/motor/delaytime',[self.id])
		return

	def GetDump(self):
		#debug("Asking motor {} for data dump".format(self.id))
		# if self.isInterfaceSerial():
		# 	cmd ="300" + str(self.id).zfill(3)
		# 	self.serial.send(cmd)
		# else:
		self.osc.sendOSC('/get/motor/dump',[self.id])
		return