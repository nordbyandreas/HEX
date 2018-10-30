import state as State
import random
import mcts as mcts

#
# Each node keeps track of relationship between nodes/states
#
#
class Node:

    #initalize node
    def __init__(self, parentNode=None, state=State.State()):
        self.parentNode = parentNode
        self.state = state
        self.childNodes = []

    def __str__(self):
        return ' ,  '.join(['( {key} = {value} )'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

    # return state of this node
    def getState(self):
        return self.state

    # sets state of this node
    def setState(self, state):
        self.state = state
    
    #return parent node of this node
    def getParent(self):
        return self.parentNode

    #sets parent node of this node
    def setParent(self, parentNode):
        self.parentNode = parentNode

    #return childnodes of this node
    def getChildren(self):
        return self.childNodes

    #sets childnodes of this node
    def setChildren(self, childNodes):
        self.childNodes = childNodes

    #add a childNode to this node
    def addChild(self, childNode=None):
        self.childNodes.append(childNode)
    
    #choose a random childnode
    def getRandomChildNode(self):
        return self.childNodes[random.randint(0, len(self.childNodes) - 1)]



    