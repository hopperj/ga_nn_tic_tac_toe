#!/bin/python

import numpy as np
from neural_network import NN
from TicTacToe import TicTacToe
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as pl
from multiprocessing import Process, Pool, Queue, Pipe
import os
from datetime import datetime, timedelta

DEBUG=0

class Player:

    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.totalGames = 0

        self.brain = NN(2,3,1)#NN(18,18*2,9)
        
        self.brain.useThresholds = 1
        self.brain.useSigmoid = 1
        

    def clearScore(self):
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.totalGames = 0
        
    def getFitness(self):
        #return 1.0 - float(self.losses)/float(self.totalGames)
        #return float(3*self.wins+2*self.ties)/float(self.totalGames)
        if self.totalGames == 0:
            return 0.0
        return float(5*self.wins+2*self.ties-10*self.losses)/float(self.totalGames)

    def maxPossibleFitness(self):
        return 5.0
    
    def lookAtBoard(self, input_board):
        if DEBUG:
            print "\n\nBoard:"        
            print input_board
        
        tmp = np.ravel( input_board )
        inputs = np.zeros( self.brain.inputs.shape )
        for i,e in enumerate(tmp):
            if e>0:
                inputs[ 2*i ] = e
            if e<0:
                inputs[ 2*i+1 ] = e
        a = self.brain.run( inputs )
        if DEBUG:
            print "Inputs:"
            print inputs,"\n"
            print a

        return a

    
def playerMove(player, game, mark):
    res = player.lookAtBoard( game.board )#player.brain.feedforward( game.board.reshape(1,9) )
    moves = [ list(res).index( e ) for e in sorted( res ) ]    
    # sort does lowest to highest, X wants highest first. O doesn't.
    """
    if mark=='x':
        #print "reversing"        
        moves.reverse()
    """
    
    if DEBUG:
        print mark
        print moves
        print "\n\n"
    
        
    foundMove = 0
    for move in moves:
        if ( not game.move( move/game.N, np.mod(move, game.M), mark ) ):
             foundMove = 1
             break
    return foundMove

def makeBaby(p):
    m = 0.01
    ihw = np.ravel(p.brain.getIHW())
    nihw = ihw + m*(np.ravel(np.zeros(p.brain.getIHW().shape))-0.5)
    nihw = nihw.reshape( p.brain.getIHW().shape )
    
    
    how = np.ravel(p.brain.getHOW())
    nhow = how + m*(np.ravel(np.random.random(p.brain.getHOW().shape))-0.5)
    nhow = nhow.reshape( p.brain.getHOW().shape )

    newPlayer = Player()
    newPlayer.brain.setIHW( nihw )
    newPlayer.brain.setHOW( nhow )

    return newPlayer
    

def mate(p1, p2):

    ihw1 = np.ravel(p1.brain.getIHW())
    ihw2 = np.ravel(p2.brain.getIHW())
    nihw = np.ravel(np.zeros(p1.brain.getIHW().shape))


    i = 0
    #for i in range( len(nihw) ):
    while i < len(nihw):
        r = np.random.random()
        n = np.random.randint(i,len(nihw)+1)
        if r<0.4:
            nihw[i:n] = ihw1[i:n]
        if r>=0.4 and r<0.8:
            nihw[i:n] = ihw2[i:n]
        if r>=0.8:
            for j in range(i,n):
                nihw[j] = 2.0*np.random.random() - 1.0
        i=n

    nihw = nihw.reshape( p1.brain.getIHW().shape )

            
    how1 = np.ravel(p1.brain.getHOW())
    how2 = np.ravel(p2.brain.getHOW())
    nhow = np.ravel(np.array(p1.brain.getHOW()))

    
    #for i in range( len(nhow) ):
    i=0
    while i < len(nhow):
        r = np.random.random()
        n = np.random.randint(i,len(nhow)+1)

        cutoffs = [ 0.49, 0.98 ]
        
        if r<cutoffs[0]:
            nhow[i:n] = how1[i:n]
        elif r<cutoffs[1]:
            nhow[i:n] = how2[i:n]
        elif r>=cutoffs[1]:
            for j in range(i,n):
                nhow[j] = np.random.random() - 0.5
        i=n

    nhow = nhow.reshape( p1.brain.getHOW().shape )


    newPlayer = Player()
    newPlayer.brain.setIHW( nihw )
    newPlayer.brain.setHOW( nhow )

    return newPlayer











def playAGame(p1, p2):
    game = TicTacToe()

    victory = [False, None]
    turnNum = 0

    result = np.zeros( (2,3) )
    
    while not victory[0]:


        #print "Player 1:"
        if not playerMove( p1, game, 'x' ):
            result[0][2] = 1
            result[1][2] = 1

            #print "No victor!!"
            #p1.ties+=1
            #p2.ties+=1            
            break

        
        #game.printBoard()
        
        turnNum+=1
        if turnNum>=5:
            victory = game.checkVictory()
        if victory[0]:
            result[0][0] = 1
            result[1][1] = 1
            #print "Player1 %c won!"%victory[1]
            #p1.wins += 1
            #p2.losses += 1
            break


        #print "Player 2:"
        if not playerMove( p2, game, 'o' ):
            result[0][2] = 1
            result[1][2] = 1
            #print "No victor!!"
            #p1.ties+=1
            #p2.ties+=1
            break

        #game.printBoard()
    
        turnNum+=1
        if turnNum>=5:
            victory = game.checkVictory()
        if victory[0]:
            result[0][1] = 1
            result[1][0] = 1
            
            #print "Player2 %c won!"%victory[1]
            #p2.wins += 1
            #p1.losses += 1
            break
        

    #print "Gamove over"
    #p1.totalGames += 1
    #p2.totalGames += 1
    
    p1.brain.reset_nodes()
    p2.brain.reset_nodes()

    return result



def tournament(matches,N,child_conn,*f):
    f=f[0]

    
    results = np.zeros((len(f),3))
    """
    # Random tournament
    for i,e in enumerate(matches):
        r1 = int( e[0]*len(f) )
        r2 = int( e[1]*len(f) )#np.random.randint(0, len(f) )
        p1 = f[ r1 ]
        p2 = f[ r2 ]

        result = np.array( playAGame(p1,p2) )

        results[r1] += result[0]
        results[r2] += result[1]
        if i%int(0.1*float(N))==0 and DEBUG:
            print "%s Completed %d of %d matches"%(os.getpid(), i,N)


    """


    """
    # Every player playes every other player N times
    for n in range(N):
        for i in range(len(f)):
            for j in range(i+1,len(f)):
                result = np.array( playAGame( f[i], f[j] ) )
                results[i] += result[0]
                results[j] += result[1]
    """

    inputs = np.array( [ [0,0], [0,1], [1,1], [1,0] ] )
    outputs = np.array( [ 0, 1, 0, 1 ] )

    for n in range(N):
        for i in range(len(f)):
            r = np.random.randint(0, 4)
            result = f[i].brain.run( inputs[r] )
            #print r,result[0]
            if result[0] == outputs[r]:
                results[i][0] += 1
            else:
                results[i][1] += 1

    
            
                       
    #q.put( results )
    child_conn.send(results)
    #print "%s Completed %d matches"%(os.getpid(),N)
    return


    
if __name__ == '__main__':
    data = []

    allstars = []
    players = []


    # Config options for 
    
    threads = 2
    gamesPerPlayer = 10
    numOfPlayers = 50
    generations = 200
    breakEarly = 1

    
    N = numOfPlayers*gamesPerPlayer/threads
    
    for i in range(numOfPlayers):
        players.append( Player() )

    try:
        for generation in range(generations):

            processes = []

            for i in range(threads):
                matches = np.random.random((N,3))
                parent_conn, child_conn = Pipe()
                p = Process( target=tournament, args=(matches,N,child_conn,players), name=str(i) )
                p.start()
                processes.append( (p, parent_conn) )
                if DEBUG:
                    print "Started process number:",i
            
            
            tournamentResults = np.zeros( (numOfPlayers,3) )

            
            t0 = datetime.now()
            while len(processes) > 0:
                for i,p in enumerate(processes):
                    if p[1].poll(1):
                        tournamentResults += p[1].recv()
                        p[0].terminate()
                        del processes[i]

            t1 = datetime.now()
            dt = (t1-t0).total_seconds()
            print "Tournament %d took %.2fs to play."%(generation,dt)
            
            fitnessReport = []
            t0 = datetime.now()    
            for i in range( numOfPlayers ):
                players[i].wins += tournamentResults[i][0]
                players[i].losses += tournamentResults[i][1]            
                players[i].ties += tournamentResults[i][2]
                players[i].totalGames += np.sum( tournamentResults[i] )

                fitnessReport.append( players[i].getFitness() )
                            
            
            fitnessReport.sort()
            assert fitnessReport[-1] >= fitnessReport[0]
            avgWinRate = np.average( fitnessReport )
            highestWinRate = np.max( fitnessReport )
            print "Average fitness: %.3f peak: %.3f generation %d"%(avgWinRate,highestWinRate,generation)
            data.append( (generation,highestWinRate) )

            

            t1 = datetime.now()
            dt = (t1-t0).total_seconds()
            print "Player update took %.2fs"%(dt)

            
            parents = []
            newPlayers = []
            wins = 0
            losses = 0
            ties = 0
            
            t0 = datetime.now()
            for i,p in enumerate(players):
                wins += p.wins
                losses += p.losses
                ties += p.ties
                if p.getFitness() == fitnessReport[-1]:
                    allStar = p,p.getFitness()
                    if allStar[1] == p.maxPossibleFitness():
                        break
                if p.getFitness() >= fitnessReport[int(len(fitnessReport)*0.75)]:
                    parents.append(p)
                if p.getFitness() >= fitnessReport[int(len(fitnessReport)*0.65)]:
                    p.clearScore()
                    newPlayers.append( p )

            t1 = datetime.now()
            dt = (t1-t0).total_seconds()
            print "Stats took %.2fs"%(dt)

    
            if allStar[1] == p.maxPossibleFitness() and breakEarly:
                break
            print "Stats\t%d\t%d\t%d\n"%(wins,losses,ties)
            children = []

            t0=datetime.now()
            while len(newPlayers) < numOfPlayers:
                #newPlayers.append( mate( parents[np.random.randint(0, len(parents) )], parents[np.random.randint(0, len(parents) )] ) )
                newPlayers.append( makeBaby( parents[np.random.randint(0, len(parents) )] ) )

            t1 = datetime.now()
            dt = (t1-t0).total_seconds()
            print "Breeding took %.2fs"%(dt)

                
            players = newPlayers
    except KeyboardInterrupt:
        pass
    print "Population of %d players"%len(players)
    print "Best cannidate fitness: %.3f"%allStar[1]
    allStar[0].brain.save()
    data = np.array( data )
    data.dump( "fitnessReport.dat" )
    pl.plot( data[:,0], data[:,1] )
    pl.title("Training error per epoch")
    #pl.show()
    pl.savefig('evolution_results')







            
