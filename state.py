
import hexboard as hb


#
#
# Every node has a state
#
# keeps track of the state of the Game AND the state of the Nodes
#
class State:
    def __init__(self, player=1, hexSize=2, hexBoard=None):

        #MCTS states
        self.visitCount = 1
        self.winCount = 0
        #MCTS states

        

        #game states
        if hexBoard is None:
            self.hexBoard = hb.HexBoard(player=player, hexSize=hexSize)
        else:
            self.hexBoard = hexBoard
        #game states

    def __str__(self):
        return ' ,  '.join(['( {key} = {value} )'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])


    #get visitcount of this node/state
    def getVisitCount(self):
        return self.visitCount

    #get wincoutn of this node/state
    def getWinCount(self):
        return self.winCount
    
    #returns the Hexboard for this state
    def getBoard(self):
        return self.hexBoard

    def setBoard(self, board):
        self.hexBoard = board


    #returns true if game is over
    def gameIsOver(self):
        return self.hexBoard.gameIsOver()


    #returns all possible next states /  all possible moves
    def getAllPossibleNextStates(self):
        return self.hexBoard.getAllPossibleNextStates()


    #gets current player
    def getPlayer(self):
        return self.hexBoard.getPlayer()


    #sets current player
    def setPlayer(self, player):
        self.hexBoard.setPlayer(player)

    # pick a random move  TODO  -  not implemented in hexmanager yet
    def randomPlay(self):
        return self.hexBoard.randomPlay()


    #increment visitcount by one
    def incrementVisitCount(self):
        self.visitCount += 1

    #increment wincount by one
    def incrementWinCount(self):
        self.winCount += 1


    #switch currentplayer from 1 to 2 and vice versa
    def switchPlayer(self, player):
        return self.hexBoard.switchPlayer(player)

