### 
# 
# main class for starting up training of the Hex player 
#
#
import numpy as np
import ANET
import node as Node
import state as State
import mcts as MCTS
import hexboard as HB

class HexTrainer():
    def __init__(self, numGames=10, hexSize=2, verbose=True, numSimulations=10, player=1):
        self.numGames = numGames
        self.hexSize = hexSize
        self.verbose = verbose
        self.numSimulations = numSimulations
        self.player = player
        self.replayBuffer = []

        self.ACM = ANET.CaseManager(self.replayBuffer)
        self.AiDim = self.hexSize * self.hexSize * 2 + 2
        self.AoDim = self.hexSize * self.hexSize
        self.AhDim = int(self.AiDim / 2)
        self.Anet = ANET.ANET([self.AiDim, self.AhDim, self.AoDim], self.ACM, learning_rate=0.1, display_interval=None, 
                                minibatch_size=10, validation_interval=None, softmax=True, error_function="ce",
                                hidden_activation_function="relu", optimizer="gradient_descent", w_range=[-0.0, 0.1],
                                grabvars_indexes=[], grabvars_types=[], lr_freq = None, 
                                bs_freq = None, early_stopping=False, target_accuracy=None)

    def run(self):

        print("Starting up..  Playing " + str(self.numGames) + " games:")

        # set save interval for actor network parameters
        # clear the replayBuffer
        # randomly init weights and biases for Actor network

        self.Anet.setupSession()


        startNode = Node.Node(state=State.State(player=self.player, hexSize=self.hexSize))
        mcts = MCTS.MCTS(numberOfSimulationsPerMove=self.numSimulations, hexTrainer = self, Anet=self.Anet)

        player = startNode.getState().getPlayer()

        startNodeCopy = startNode

        player1Wins = 0
        player2Wins = 0
        player1Starts = 0
        player2Starts = 0

        gc = 1

        #for a game in numberOfGames
        for game in range(0, self.numGames):
            #Start of a game

            #clear replay buffer
            self.replayBuffer = []


            #initialize gameboard to empty board
            startNode = startNodeCopy
            startingPlayer = startNode.getState().getPlayer()

            if startingPlayer == 1:
                player1Starts += 1
            else: 
                player2Starts += 1

            print("\n\n\n --- Game number " + str(gc))

            #print starting state
            startNode.getState().getBoard().printBoard()
        
            while not startNode.getState().gameIsOver():
    
                player = startNode.getState().getPlayer()
                
                
                #use tree policy to search from root to leaf
                #use ANET to choose rollout actions from L to final state
                #perform mcts-backpropogation
                #next gamestate
                print("Player " + str(player) + "'s turn")
                print("legal moves:")
                print(startNode.getState().getBoard().getLegalMoves())
                
                
                nextNode = mcts.findNextMove(startNode, player, startingPlayer)

                # D = distribution of visitCounts alogn all arcs emanating from root
                # add case (root, D) to replayBuffer
                #choose actual move (action*) based on D
                #perform action* on root to produce successor state s*
                #update currentstate to s*
                # in mcts - retain subtree rooted at s*, discard everything else
                # rootnode = s*

                #TODO change this ? 
                if self.verbose: nextNode.getState().getBoard().printBoard()

                if nextNode.getState().gameIsOver():

                    if self.verbose: print("\nPlayer " + str(player) + " won! \n")
                    
                    if player == 1: 
                        player1Wins += 1
                    else:
                        player2Wins += 1

                startNode = nextNode
                if nextNode.getState().gameIsOver():
                    break
            gc += 1
            
            # train ANET on random minibatch of cases from replayBuffer
            np.random.shuffle(self.replayBuffer)
            inputs = [case[0] for case in self.replayBuffer]; targets = [case[1] for case in self.replayBuffer] 
            print("inputs:")
            print(inputs)
            print("targets")
            print(targets)
            feeder = {self.Anet.input: inputs, self.Anet.target: targets}
            gvars = [self.Anet.error] + self.Anet.grabvars
            self.Anet.run_one_step([self.Anet.trainer], gvars, session=self.Anet.current_session, feed_dict=feeder)
                       
            
            # if gameNum %modulo saveinterval: save ANET parameters for later use in TOPP



            #next game  


        #print result of all games
        print("\nPlayer 1 started " + str(player1Starts) + " games and won " + str(player1Wins) + " of "  + str(self.numGames) + " games!   " + str((player1Wins/self.numGames*100)) + " % ")
        print("Player 2 started " + str(player2Starts) + " games and won " + str(player2Wins) + " of "  + str(self.numGames) + " games!   " + str((player2Wins/self.numGames*100)) + " % ")
        print("\n")


        self.Anet.close_current_session(view=False)











# entry for starting the training
def main():
   trainer = HexTrainer(numGames=1, hexSize=3, verbose=True, numSimulations = 1, player=1)
   trainer.run()
   print(trainer.replayBuffer)

if __name__ == '__main__':
    main()
 