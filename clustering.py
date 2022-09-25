# CSCI 460 Clustering Project
# Nick Diaz

from pandas import read_csv
import numpy as np
from random import randint
import sklearn.metrics
import matplotlib.pyplot as plt

# Expectation Maximization algorithm
def ExpectationMax(data, k, sd):

    # Initialize hypothesis with k random points
    # from data
    h = np.empty((k,2), float)
    for i in range(k):
        h[i] = data[randint(0,29)]

    print('-'*80)
    print("Initial hypothesis:\n", h)
    print()
    iteration = 0

    # E-Step
    assignments = [0 for x in range(30)]    # list that will hold each point's cluster assignmet
    while iteration < 20:
        # Don't uncomment this print() unless other lines below are uncommented;
        # will just be displaying a ton of lines for no reason
        #print('-'*80)     # Indicates beginning of a new iteration
        expProbs = np.empty((30,k), float)
        for i in range(30):
            for j in range(k):
                # Calculating denominator
                denom = 0.0
                for n in range(k):
                    denom += np.exp(-(1/(2*sd))*(np.linalg.norm(data[i] - h[n])))
                # Equation from slides
                expProbs[i, j] = ( np.exp(-(1/(2*sd))*(np.linalg.norm(data[i] - h[j]))) ) / denom

            # Assign point i with its respective greatest probability in expProbs
            assignments[i] = expProbs.tolist()[i].index(max(expProbs[i]))
                              
        #print("Expected Probabilities for Z(i,k):\n", expProbs)
        #print("Assignments:\n", assignments)

        # M-Step
        for j in range(k):
            for i in range(30):
                # Calculate numerator and denominator first and separately
                numer = np.empty((1,2), float)
                denom = 0.0
                for a in range(30):
                    numer[0] += data[a] * expProbs[a,j]
                    denom += expProbs[a,j]
                # Equation from slides
                h[j] =  numer / denom
        iteration += 1

        #print("Iteration: ", iteration)
        #print("Updated hypothesis:\n", h)
        #print()

    print("Final hypothesis:\n", h)
    print()
    print("Final assignments:\n", assignments)
    print()
    print("Bouldin Score for k =", k,"and Standard Deviation:", sd, ":\n", sklearn.metrics.davies_bouldin_score(data, assignments))



# Read in the dataset
dataset = read_csv('pr3.data')
data = np.array(dataset)

plt.scatter(dataset['X'], dataset['Y'])
plt.show()


# Run the algorithm several times with different
# k and standard deviation values
# k = 2
ExpectationMax(data, 2, 2.0)
ExpectationMax(data, 2, 1.0)
ExpectationMax(data, 2, 0.5)
# k = 3
ExpectationMax(data, 3, 2.0)
ExpectationMax(data, 3, 1.0)
ExpectationMax(data, 3, 0.5)
# k = 4
ExpectationMax(data, 4, 2.0)
ExpectationMax(data, 4, 1.0)
ExpectationMax(data, 4, 0.5)