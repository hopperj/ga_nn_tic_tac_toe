


from multiprocessing import Process, Pool, Queue
import numpy as np

class A:

    def __init__(self):
        self.a = 0
        self.b = 0


def tournament(q,matches,*f):
    n=10
    f=f[0]

    N=100
    
    results = np.zeros((len(f),3))
    
    for e in matches:
        r1 = int( e[0]*len(f) )
        r2 = int( e[1]*len(f) )#np.random.randint(0, len(f) )
        p1 = f[ r1 ]
        p2 = f[ r2 ]

        result = np.array( playGame(p1,p2) )

        results[r1] += result[0]
        results[r2] += result[1]
        
    q.put( results )

def playGame(p1, p2):
    r = np.random.randint(0,4)
    result = np.zeros((2,3))
    if r==1:
        result[0][0] = 1
        result[1][1] = 1
    if r==2:
        result[0][1] = 1
        result[1][0] = 1
    if r==3:
        result[0][2] = 1
        result[1][2] = 1
    return result
    
    
objs = []
for i in range(20):
    objs.append( A() )

import numpy as np

"""
fighters = []
for i in range(30):
    fighters.append( [objs[0], objs[1]] )
    #fighters.append( (objs[ int(np.random.rand()*len(objs)) ], objs[ int(np.random.rand()*len(objs)) ]) )
"""
    
processes = []
q = Queue()
N = 100
for i in range(5):
    matches = np.random.random((N,3))
    p = Process( target=tournament, args=(q,matches,objs) )
    p.start()
    processes.append( p )

for p in processes:
    p.join()

#for obj in objs:
#    print obj.a,obj.b
masterResults = np.zeros( (len(objs), 3 ) )
while not q.empty():
    masterResults += q.get()
print masterResults,"\n\n"
