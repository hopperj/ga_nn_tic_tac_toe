#!/bin/python

import numpy as np




class TicTacToe:

    def __init__(self, N=3, M=3):
        self.board = np.zeros((N,M))
        self.N = N
        self.M = M
        self.tolkens = { 'x':1, 'o':-1 }

    def getBoard(self):
        return self.board

    def validMove(self, x,y ):
        if x>=self.N or y>=self.M:
            # returning 0 indicates it is not valid
            return 0
        return self.board[x,y] == 0

    def move(self, x, y, mark):
        if self.validMove(x,y):
            self.board[x,y] = self.tolkens[mark]            
            # 0 indicates no error
            return 0
        # return 1 indicating error
        return 1

    def printBoard(self):
        #print "\n\n"
        for i in range(self.N):
            print "-------------"
            line = "|"
            for j in range(self.M):
                if self.board[i,j] == 1:
                    line += " x |"
                if self.board[i,j] == -1:
                    line += " o |"
                if self.board[i,j] == 0:
                    line += "   |"
            print line
        print "-------------\n\n"
            
    def printRawBoard(self):
        #print "\n\n"
        for i in range(self.N):
            print "-------------"
            line = "|"
            for j in range(self.M):
                if self.board[i,j] == -1:
                    line += " -1|"
                if self.board[i,j] == 1:
                    line += " 1 |"
                if self.board[i,j] == 0:
                    line += " 0 |"
            print line
        print "-------------\n\n"
            
    def checkVictory(self):
        if np.all(self.board.T>0,-1).any() or np.all(self.board>0,-1).any() or np.diagonal(self.board>0).all() or np.diagonal(np.flipud(self.board>0)).all():
            #print "O won"
            return True,'x'

        if np.all(self.board.T<0,-1).any() or np.all(self.board<0,-1).any() or np.diagonal(self.board<0).all() or np.diagonal(np.flipud(self.board<0)).all():
            #print "X won"
            return True,'o'

        return False,None

        # check the diagonals
        #return np.diagonal(self.board).all() or np.diagonal(np.flipud(self.board)).all()
            




if __name__ == '__main__':

    game = TicTacToe()

    print "Checking Reverse Diagonal"
    game = TicTacToe()
    game.board[0,2] = 1
    game.board[1,1] = 1
    game.board[2,0] = 1
    assert game.checkVictory() == True

    print "Checking Diagonal"
    game = TicTacToe()
    game.board[0,0] = -1
    game.board[1,1] = -1
    game.board[2,2] = -1
    assert game.checkVictory() == True

    print "Checking no win senario"
    game = TicTacToe()
    game.board[0,0] = -1
    game.board[1,1] = 1
    game.board[2,2] = -1
    assert game.checkVictory() == False


    print "Checking column win"
    game = TicTacToe()
    game.board[0,1] = -1
    game.board[1,1] = -1
    game.board[2,1] = -1
    assert game.checkVictory() == True

    print "Checking row win"
    game = TicTacToe()
    game.board[1,0] = -1
    game.board[1,1] = -1
    game.board[1,2] = -1
    assert game.checkVictory() == True

    print "Check non diagonal win"
    game = TicTacToe()
    game.board[0,0] = -1
    game.board[1,1] = 1
    game.board[2,2] = -1
    assert game.checkVictory() == False



    game.board[1,0] = -1
    game.board[2,2] = -1

    print "Victor?",game.checkVictory()
    game.printBoard()
