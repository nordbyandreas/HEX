
import math
import random



#
#
#
# HexBoard class - acts the role of the StateManager referred to in the document
#
class HexBoard:
    def __init__(self, player=1, hexSize = 2, legalMoves=None):
        self.hexSize = hexSize
        self.board = self.buildBoard(hexSize)
        self.whiteSideLeft, self.whiteSideRight, self.blackSideLeft, self.blackSideRight = self.getHexSides()
        self.winner = None
        self.player = player
        
        if legalMoves is None:
            self.legalMoves = [1] * (hexSize * hexSize)
        else:
            self.legalMoves = legalMoves

    #get a list of possible positions on the 'board' and play a random move
    def randomPlay(self):
        nextStates = self.getAllPossibleNextStates()
        randomChoice = nextStates[random.randint(0, len(nextStates) - 1)]
        return randomChoice
    
    def getLegalMoves(self):
        return self.legalMoves

    def getPlayer(self):
        return self.player

    def setPlayer(self, player):
        self.player = player

    def copyBoardValues(self, boardToCopy):
        for i, row in enumerate(boardToCopy, 0):
            for j, col in enumerate(row, 0):
                self.board[i][j].setValue(boardToCopy[i][j].getValue())
      

    #TODO: how do dis ?
    # too much work to copy entire boardobject for every new state?
    # instead, keep a 

    # returns a list of all possible next states
    def getAllPossibleNextStates(self):
        nextStates = []
        for i, row in enumerate(self.board, 0):
            for j, col in enumerate(row, 0):
                if col.getValue() == 0:
                 
                    boardCopy = HexBoard(player=(3-self.player), hexSize=self.hexSize, legalMoves=self.legalMoves.copy())
                    boardCopy.copyBoardValues(self.board)
                    
                    if self.player == 1:
                        boardCopy.getBoard()[i][j].setValue(1)
                    else:
                        boardCopy.getBoard()[i][j].setValue(2)
                    #remove one legal move 
                    moveIndex = i*self.hexSize + j
                  
                 
                    boardCopy.legalMoves[moveIndex] = 0
                

                    nextStates.append(boardCopy)
        return nextStates





    # return winner
    def getWinner(self):
        return self.winner

    # returns true if the game is over and determines the winner (sets self.winner)
    def gameIsOver(self):
        unvisitedWhiteNodes = []; unvisitedBlackNodes = []
        visitedWhiteNodes = []; visitedBlackNodes = []
        for node in self.whiteSideLeft:
            if node.getValue() == 1: unvisitedWhiteNodes.append(node)
        while unvisitedWhiteNodes:
            checkNode = unvisitedWhiteNodes[0]
            for neighbor in checkNode.getNeighbors():
                if neighbor.getValue() == 1 and neighbor not in unvisitedWhiteNodes and neighbor not in visitedWhiteNodes:
                    unvisitedWhiteNodes.append(neighbor)
            visitedWhiteNodes.append(unvisitedWhiteNodes.pop(0))
            if checkNode in self.whiteSideRight:
                #print("white won")
                self.winner = 1
                return True
        
        for node in self.blackSideLeft:
            if node.getValue() == 2: unvisitedBlackNodes.append(node)
        while unvisitedBlackNodes:
            checkNode = unvisitedBlackNodes[0]
            for neighbor in checkNode.getNeighbors():
                if neighbor.getValue() == 2 and neighbor not in unvisitedBlackNodes and neighbor not in visitedBlackNodes:
                    unvisitedBlackNodes.append(neighbor)
            visitedBlackNodes.append(unvisitedBlackNodes.pop(0))
            if checkNode in self.blackSideRight:
                #print("black won")
                self.winner = 2
                return True

        return False
        



    #defines which sides of the hexboard that belong to each player
    # returns the nodes for each side
    def getHexSides(self):
        #white will get (0,0), (1, 0), (2,0)  and (0, n), (1, n), (2, n) etc
        whiteNodesLeft = []
        whiteNodesRight = []
        #black will get (0, 0), (0, 1), (0, 2)  and (n, 0), (n, 1), (n, 2) etc
        blackNodesLeft = []
        blackNodesRight = []

        for i in range(0, self.hexSize):
            whiteNodesLeft.append(self.board[i][0]); whiteNodesRight.append(self.board[i][self.hexSize - 1])
            blackNodesRight.append(self.board[0][i]); blackNodesLeft.append(self.board[self.hexSize - 1][i]); 

        return whiteNodesLeft, whiteNodesRight, blackNodesLeft, blackNodesRight

        
    #returns the board
    def getBoard(self):
        return self.board

    def switchPlayer(self, player):
        return 3 - player

    # returns the vector-representation of the boardstate
    def getNeuralRepresentation(self, player):
        neuralRepr = []
        board = self.board
        for row in board:
            for col in row:
                if col.getValue() == 1:
                    neuralRepr.append(float(1)); neuralRepr.append(float(0))
                elif col.getValue() == 2:
                    neuralRepr.append(float(0)); neuralRepr.append(float(1))
                else:
                    neuralRepr.append(float(0)); neuralRepr.append(float(0))
        if player == 1:
            neuralRepr.append(float(1)); neuralRepr.append(float(0))
        else:
            neuralRepr.append(float(0)); neuralRepr.append(float(1))
        return neuralRepr


    #builds the hexboard
    def buildBoard(self, hexSize):
        board = []
        for i in range(0, hexSize):
            row = []
            for j in range(0, hexSize):
                row.append(HexCell())
            board.append(row)

        # create connections between cells (neighbor relationship)
        self.connectBoard(board)
        return board

    # goes through every cell in the board and connects them to their neighbors
    def connectBoard(self, board):
        for i, row in enumerate(board, 0):
            for j, cell in enumerate(row, 0):
                #add all possible neighboors
                self.addCellNeighbors(cell, board, i, j)

    # add neighbors to a single cell
    def addCellNeighbors(self, cell, board, rowIndex, columnIndex):
        maxIndex = self.hexSize - 1
        if rowIndex-1 >= 0:
            cell.addNeighbor(board[rowIndex-1][columnIndex])
      
        if rowIndex -1 >= 0 and columnIndex + 1 <= maxIndex:
            cell.addNeighbor(board[rowIndex-1][columnIndex+1])

        if columnIndex - 1 >= 0:
            cell.addNeighbor(board[rowIndex][columnIndex - 1])

        if columnIndex + 1 <= maxIndex:
            cell.addNeighbor(board[rowIndex][columnIndex + 1])

        if rowIndex + 1 <= maxIndex and columnIndex - 1 >= 0:
            cell.addNeighbor(board[rowIndex +1][columnIndex - 1])

        if rowIndex + 1 <= maxIndex:
            cell.addNeighbor(board[rowIndex + 1][columnIndex])
    
    def coordinateIsInBoard(self, iRow, iCol):
        maxIndex = self.hexSize - 1
        return iRow >= 0 and iRow <= maxIndex and iCol >= 0 and iCol <= maxIndex

    #prints the board
    def printBoard(self):
        board = self.board
        maxIndex = self.hexSize - 1
        lines = []
        metaColIndex = 0
        metaRowIndex = -1

        # organize board into hex shape
        for i in range(0, self.hexSize*2 - 1):
            if i > maxIndex: metaColIndex += 1
            if i <= maxIndex: metaRowIndex += 1
            rowIndex = metaRowIndex
            colIndex = metaColIndex
            line = []
            while self.coordinateIsInBoard(rowIndex, colIndex):
                line.append(board[rowIndex][colIndex])
                rowIndex -= 1
                colIndex += 1
            lines.append(line)

        stringSpace = "   "
        spaceOffset = "                  "
        spaceController = self.hexSize
        spaceDecrease = True
        print(stringSpace + spaceOffset + "------"*maxIndex)
        print(stringSpace + spaceOffset + "W" + "------"*maxIndex + "B")

        #actually print board in hex format
        for i in range(0, self.hexSize*2 - 1):
            space = spaceOffset + stringSpace * spaceController
            for cell in lines[i]:
                printValue = "?"
                if cell.getValue() == 1:
                    printValue = "W"
                if cell.getValue() == 2:
                    printValue = "B"
                if cell.getValue() == 0:
                    printValue = "O"
           
                space += printValue + "     "

            print(space)

            #control when to switch from removing space to adding space
            if i >= maxIndex: spaceDecrease = False
            
            #add or remove space before printing values
            if spaceDecrease:
                spaceController -= 1
            else:
                spaceController += 1

        print(stringSpace + spaceOffset + "B" + "------"*maxIndex + "W")
        print(stringSpace + spaceOffset + "------"*maxIndex)
        



            








            
        

#
# Class representing a single cell in the HexBoard
#
class HexCell:
    def __init__(self, value=0):
        self.value = value
        self.neighbors = []

    def setValue(self, value):
        self.value = value

    def getValue(self):
        return self.value

    def addNeighbor(self, cell):
        self.neighbors.append(cell)

    def getNeighbors(self):
        return self.neighbors
    
    

