'''
#------------------------------------------------------------
  filename: lab1_randomwalk.py
  This is an example for random walk problem using numpy ndarray

  written by Jaewook Kang @ Aug 2017
#------------------------------------------------------------
'''
import numpy as np

position = 0
next_pos = 0
step = 10

randwalk = np.zeros(step)

for i in range(step):
    coin = np.random.randint(2,size=1)
    randwalk[i] = position

    if coin == 1:
        print 'Coin is 1'
        next_pos = position + 1
    else:
        print 'Coin is 0'
        next_pos = position - 1
    position = next_pos


print 'My randomWalk is %s ' % randwalk






