
import node as Node
import math
import state as State
import hexboard as HB
from sklearn import preprocessing


# Monte Carlo Tree Search class
#
# Basically in charge of choosing and learning.
#
# Finds the next move and updates the Q values in the search tree while playing
#
class MCTS:
    def __init__(self, numberOfSimulationsPerMove = 5, hexTrainer=None, Anet=None):
        self.numberOfSimulationsPerMove = numberOfSimulationsPerMove
        self.hexTrainer = hexTrainer
        self.Anet = Anet


    def findNextMove(self, node, player, startingPlayer):   
        
        rootNode = node
        rootNode.getState().setPlayer(player)


        simulations = self.numberOfSimulationsPerMove

        while simulations > 0:
     
            #SELECTION
            selectedNode = self.selectNode(rootNode, player, startingPlayer)
            #print("after selectNode")

            #EXPANSION
            selectedNode = self.expandNode(selectedNode)
       

            #SIMULATION
            if len(selectedNode.getChildren()) > 0:
                selectedNode = selectedNode.getRandomChildNode()

            winner = self.simulateRandomPlayout(selectedNode)
       


            #UPDATE (backpropagate)
            self.backpropagate(selectedNode, winner, startingPlayer)
      

            simulations = simulations - 1
      


        # add case (rootState, D) to replayBuffer
        self.addToReplayBuffer(rootNode)

    

        selectedMove = chooseMostVisitedPath(rootNode, player, startingPlayer)

        return selectedMove
 

    # D = distribution of visitCounts alogn all arcs emanating from root
    # add case (root, D) to replayBuffer
    def addToReplayBuffer(self, node):
        inp = node.getState().getBoard().getNeuralRepresentation(node.getState().getPlayer())
        children = node.getChildren()
        Dpre = []
        lMoves = node.getState().getBoard().getLegalMoves()
        for node in children:
            Dpre.append(node.getState().getVisitCount())
        D = [0]*(node.getState().getBoard().hexSize * node.getState().getBoard().hexSize)
        print("\n DEBUG")
        print(lMoves)
        print(D)
        print(Dpre)
        for move in range(0, len(lMoves)):
            if lMoves[move] == 1:
                D[move] = Dpre.pop(0) 
        print("\n")
        print(lMoves)
        print(D)
        print(Dpre)
        print("DEBUG\n")
        Dnorm = [float(i)/sum(D) for i in D]
        
        
        
        print("neural rep. of rootnode state:")
        print(inp)
        print("probs")
        print(Dnorm)
        case = [inp, Dnorm]
        self.hexTrainer.replayBuffer.append(case)
        

        



    # simulates a game until game over and returns the winner.
    # uses the 'default policy' which in this case is random picks 
    def simulateRandomPlayout(self, node):
        currentPlayer = node.getState().getPlayer()
        tempNode = node
        tempState = tempNode.getState().getBoard()
    
        if(tempState.gameIsOver()):
            #winner was the one who took the action that led to game over
            winner = 3 - currentPlayer
            return winner
         
        while not tempState.gameIsOver():
            if (tempState.gameIsOver()):
                winner = 3 - tempState.getPlayer()
                return winner
        
            tempState = self.AnetPolicy(tempState)
            #tempState = tempState.randomPlay()
            
            # use default policy to make moves here 
            tempState.switchPlayer(tempState.getPlayer())
    
    # chooses next state based on actor network's choice
    def AnetPolicy(self, state):
        neuralState = state.getNeuralRepresentation(state.getPlayer())
        feeder = {self.Anet.input: [neuralState]}
   
        legalMoves = state.getLegalMoves()
      
        AnetOutput = self.Anet.current_session.run(self.Anet.output, feed_dict=feeder)
        AnetOutput = AnetOutput[0]
       
        for i in range(0, len(AnetOutput)):
            AnetOutput[i] = AnetOutput[i] * legalMoves[i]
        AnetOutput = [float(i)/sum(AnetOutput) for i in AnetOutput]
        nextState = HB.HexBoard(player=(3-state.player), hexSize=state.hexSize, legalMoves=state.legalMoves.copy())
        nextState.copyBoardValues(state.getBoard())
        index = AnetOutput.index(max(AnetOutput))
        i = int(index / (state.hexSize))
        j = index % (state.hexSize)
        nextState.getBoard()[i][j].setValue(state.getPlayer())
        nextState.legalMoves[index] = 0
        #print("session run output after removal::")
        #print(AnetOutput)
        return nextState

    # propagate the result back upwards to all parent states/nodes
    def backpropagate(self, selectedNode, winner, startingPlayer):
        tempNode = selectedNode
        while tempNode != None:
            tempNode.getState().incrementVisitCount()
   
            #update wincount for states
            if(tempNode.getState().getPlayer() != winner):
                tempNode.getState().incrementWinCount()
            tempNode = tempNode.getParent()
       

    # selects next node based on the 'tree policy' - uses UCB1 function to determine values 
    def selectNode(self, rootNode, player, startingPlayer):
        node = rootNode
        if node is not None:
            while len(node.getChildren()) is not 0:
                newNode = UCT(node, player, startingPlayer, True)
                node = newNode
        return node

    #expands chosen node (adds childnodes of possible next states)
    def expandNode(self, node):
        possibleNextStates = node.getState().getAllPossibleNextStates()
        tempNode = node
        for state in possibleNextStates:
            childNode = Node.Node(state=State.State(hexBoard=state))
            childNode.setParent(tempNode)
            tempNode.addChild(childNode)
           
        return tempNode




#misleading name, actually chooses based on wincount/visitcount
def chooseMostVisitedPath(node, player, startingPlayer):
    children = node.getChildren()
    temp = -999999999
    choice = None

    for child in children:
        if child.getState().getWinCount() / child.getState().getVisitCount() >= temp:
            temp = child.getState().getWinCount() / child.getState().getVisitCount()
            choice = child
        

    return choice



# UCT function to determine values in Selection phase.
#  uses the "upper confidence bound" method
# 
def UCT(node, player, startingPlayer, rollout):
    parentVisits = node.getState().getVisitCount()
    explorationParam = 1 #math.sqrt(2)  'c'
    
    if parentVisits == 0:
        parentVisits = parentVisits + 1

    children = node.getChildren()
    selectedNode = None
    if player == startingPlayer:
        tempMax = -9999999
        #choose max
     
        for childNode in children:
            exploitationTerm = childNode.getState().getWinCount() / (childNode.getState().getVisitCount() + 1)
            
            #ucb1 function
            value = exploitationTerm +  explorationParam * math.sqrt((math.log(parentVisits))/(1 + childNode.getState().getVisitCount()))
           
            if value >= tempMax:
                tempMax = value
                selectedNode = childNode
    else:
        tempMax = 99999999
        #choose min
        for childNode in children:
            exploitationTerm = childNode.getState().getWinCount() / (childNode.getState().getVisitCount() + 1)
            
            #ucb1 function
            value = exploitationTerm +  explorationParam * math.sqrt((math.log(parentVisits))/(1 + childNode.getState().getVisitCount()))
      
            if value <= tempMax:
                tempMax = value
                selectedNode = childNode
    return selectedNode