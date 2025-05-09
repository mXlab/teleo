# def onStateEnter/Exit__$Statename
# def onTransitionStart/End__$Surce__$Target
# def onStateCycle__$Statename


def onStateEnter(stateName:str, prevState:str, transitionTime:float, dataStore:COMP, stateMachine:"extBananaMash"):
    if stateName == 'Sleeping':
        op('Indexer').par.const0value = 0
        #op('switch1').par.index = 0
        #op('OSC_State/osc_state_sender').sendOSC('/state', 'Sleeping')


    elif stateName == 'Waking':
        op('Indexer').par.const0value = 1
        #op('switch1').par.index = 1
        #op('OSC_State/osc_state_sender').sendOSC('/state', 'Waking')

    elif stateName == 'Exploring':
        op('Indexer').par.const0value = 2
        #op('switch1').par.index = 2
        #op('OSC_State/osc_state_sender').sendOSC('/state', 'Exploring')

    elif stateName == 'Socializing':
        op('Indexer').par.const0value = 3
        #op('switch1').par.index = 3
        #op('OSC_State/osc_state_sender').sendOSC('/state', 'Socializing')

    elif stateName == 'Hiding':
        op('Indexer').par.const0value = 4
        #op('switch1').par.index = 4
        #op('OSC_State/osc_state_sender').sendOSC('/state', 'Hiding')

    elif stateName == 'Aggressive':
        op('Indexer').par.const0value = 5
        #op('switch1').par.index = 5
        #op('OSC_State/osc_state_sender').sendOSC('/state', 'Aggressive')
    return

def onStateExit( stateName:str, nextState:str, transitionTime:float, dataStore:COMP, stateMachine:"extBananaMash"):
    return

def onTransitionStart( fromState:str, toState:str, transtionTime:float, dataStore:COMP, stateMachine:"extBananaMash"):
    return

def onTransitionEnd( fromState:str, toState:str, transtionTime:float, dataStore:COMP, stateMachine:"extBananaMash"):
    return

def onStateCycle(stateName, dataStore:COMP, stateMachine:"extBananaMash"):
    return