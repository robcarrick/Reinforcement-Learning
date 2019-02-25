### K-Armed Bandit
# Replication of p.29 of Reinforcement Learning: An Introduction,
# (Sutton and Barto, 2018)

# =============================================================================
# Import libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Create the bandit algorithm
# =============================================================================
def bandit(K,steps,eps,initialVal,UCB):
    # INPUT: K - number of bandits
    #        steps - number of updating steps 
    #        eps - probability of taking a random action (for exploration)
    #        initialVal - initial Q values
    #        UCB - setting to 0 will implement eps-greedy algorithm
    #              setting to >0 will implement UCB algorithm with UCB being
    #              the degree of exploration parameter
    
    # OUTPUT: will output an array of the rolling mean reward and
    #         the best expected reward value  
    
    # error catching
    if UCB < 0:
        print('UCB argument must be >= 0')
        return
    
    np.random.seed(200)
    # parameter initialisation
    means = np.random.normal(0,1,K) # Means for each bandit
    bestExpVal = max(means)
    Q = np.zeros(K) + initialVal
    N = np.zeros(K, dtype=int) # initialise Q values and counts
    
    RHistory = np.zeros(steps)
    for i in range(steps):
        
        if UCB == 0: # choose action according to eps-greedy algorithm
            unif = np.random.uniform()
            if unif < eps:
               A = np.random.randint(0,K) 
            else:
                argmaxInd = np.where(Q == max(Q))[0]
                A = np.random.choice(argmaxInd,1)
        else: # choose action according to UCB method
            if 0 in N:
                zeroInd = np.where(N == 0)[0]
                A = np.random.choice(zeroInd,1)
            else:
                UCBArray = Q + UCB*np.sqrt(np.log(i+1)/N)
                argMaxInd = np.where(UCBArray == max(UCBArray))[0]
                A = np.random.choice(argMaxInd,1)
                
        R = np.random.normal(means[A],1,1) # sample reward from the chosen action
        # update rolling mean of rewards
        if i == 0 :
            RHistory[i] = R
        else:
            RHistory[i] = RHistory[i-1] + 1/(i+1) * (R-RHistory[i-1])
        # update state visit count and the Q value for chosen action A
        N[A] += 1
        Q[A] += 1/N[A] * (R-Q[A])
        
    print('\nEpsilon = ', eps)
    print('Means: ',means)
    print('\nEstimated means: ',Q)
    print('\nTrials for each bandit: ',N)
    
    return([RHistory,bestExpVal])

# =============================================================================
# Run the algorithm and plot average reward curves
# =============================================================================
b1 = bandit(K=10, steps=10000, eps=0.1, initialVal=0, UCB=0)
b2 = bandit(K=10, steps=10000, eps=0.01, initialVal=0, UCB=0)
b3 = bandit(K=10, steps=10000, eps=0, initialVal=0, UCB=0) # greedy

# plot average reward curves for each epsilon
plt.plot(b1[0], label='eps=0.1')
plt.plot(b2[0], label='eps=0.01')
plt.plot(b3[0], label='eps=0')
plt.xlabel('Steps'); plt.ylabel('Average Reward')
plt.legend()
plt.show()
print('Optimal action mean reward = ', b1[1])

