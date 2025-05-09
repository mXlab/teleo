# me - this DAT
# 
# channel - the Channel object which has changed
# sampleIndex - the index of the changed sample
# val - the numeric value of the changed sample
# prev - the previous sample value
# 
# Make sure the corresponding toggle is enabled in the CHOP Execute DAT.

def onOffToOn(channel, sampleIndex, val, prev):
	return

def whileOn(channel, sampleIndex, val, prev):
	return

def onOnToOff(channel, sampleIndex, val, prev):
	return

def whileOff(channel, sampleIndex, val, prev):
	return

def onValueChange(channel, sampleIndex, val, prev):
	
	ops = [op('fisrt_joint'),op('second_joint'),op('head_joint'),op('head')]
	idx = channel.index
	
	ops[idx].par.Jointgoal = val
	ops[idx].par.Jointspeed = op('speed')[0]

	
	
	
	return
	