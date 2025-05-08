# def onStateEnter/Exit__$Statename
# def onTransitionStart/End__$Surce__$Target
# def onStateCycle__$Statename


def onStateEnter(stateName: str, prevState: str, transitionTime: float, dataStore: COMP, stateMachine: "extBananaMash"):
    if stateName == 'Sleeping':
        op('switch1').par.index = 0

    elif stateName == 'Waking':
        op('switch1').par.index = 1

    elif stateName == 'Exploring':
        op('switch1').par.index = 2

    elif stateName == 'Socializing':
        op('switch1').par.index = 3

    elif stateName == 'Hiding':
        op('switch1').par.index = 4

    elif stateName == 'Aggressive':
        op('switch1').par.index = 5

    return

def onStateExit( stateName:str, nextState:str, transitionTime:float, dataStore:COMP, stateMachine:"extBananaMash"):
    return

def onTransitionStart( fromState:str, toState:str, transtionTime:float, dataStore:COMP, stateMachine:"extBananaMash"):
    return

def onTransitionEnd( fromState:str, toState:str, transtionTime:float, dataStore:COMP, stateMachine:"extBananaMash"):
    return

def onStateCycle(stateName, dataStore:COMP, stateMachine:"extBananaMash"):
    return