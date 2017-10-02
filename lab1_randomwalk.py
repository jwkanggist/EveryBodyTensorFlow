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


# algo 1

# for i in range(step):
#     coin = np.random.randint(2,size=1)
#     randwalk[i] = position
#
#     if coin == 1:
#         print 'Coin is 1'
#         next_pos = position + 1
#     else:
#         print 'Coin is 0'
#         next_pos = position - 1
#     position = next_pos
#
#

# algo 2
randwalk = np.zeros(step)
coin_toss_results = 2 * np.random.randint(2,size=step) - 1
randwalk = np.cumsum(coin_toss_results)

negative_value_index = np.where(randwalk < 0)
print '# My randomWalk is %s ' % randwalk

print '# Negative values in randomWalk are located at index %s' % negative_value_index
print '# Whose values are randwalk[%s] = %s' % (negative_value_index ,randwalk[negative_value_index])





