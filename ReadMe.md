# CSCI 460 Clustering Project

Please implement the **Expectation Maximization** based k-means clustering algorithm we discussed in class and apply it to the data set provided (*pr3.data*) for k ∈ {2, 3, 4}.  Consider various values for σ, including σ ∈ {0.5, 1.0, 2.0}, and run the clustering multiple times for each σ value under different initial conditions. Show these results. Evaluate the quality of the clustering for each value of σ using an average of the Davies-Bouldin Index for each clustering.

In addition to providing the code and results just described, please answer the following questions:
*  How sensitive is the method to σ?
*  Which value for σ appeared work the best?
*  How sensitive is the method to k?
*  Which value for k appeared to work the best?

Please submit your answers in the BlackBoard assignment submission field text box, but I will grade your source code by pulling from your GitHub repository. So make sure it is pushed by the due date.

Please make sure your code is documented sufficiently so that it is easy to know how read and execute your program. Do not collaborate with other students, and do not use code off of the Internet (other than what I give you).

## Reading And Dealing with the Data
You may read in the data in whatever way you like; however, I suggest using the *Pandas* package.  I also suggest using Numpy to deal with vectors.  Some code examples below may be useful to you:

```
import pandas as pd
import numpy as np

# Load the data:
pr3 = pd.read_csv('pr3.data')

# Get all values in column 1:
pr3['X']

# Get values associated with row 1:
pr3.iloc[0]

# Convert the whole dataset to a Numpy matrix:
np.array(pr3)

# Use Numpy to compute the L2 norm distance between two points:
x = np.array(pr3.iloc[1])
z = np.array(pr3.iloc[0])
np.linalg.norm(x-z)

# Use Numpy to compute stats over data
m, d = np.shape(pr3)  # Get the size and dimensionality of the dataset
np.sum(pr3['X'])  # Sum of the X column
np.mean(pr3['X']) # Average of the X column
np.std(pr3['X'])  # Standard deviation of the X column

# Numpy's random module may be helpful
np.random.choice(range(m), 3, replace=False)  # Choose three 1:m w/o replacement
np.random.normal(loc=2.1, scale=0.2, size=4)  # Draw four numbers from N(2.1, 0.2)

# Numpy has an exp function.  So you can compute e^(2) as follows:
np.exp(2)
```



## Do Your Own Coding
To be clear:  You are responsible for implementing this.  You *cannot* coordinate or communicate with others or copy code off the Internet.  This is *your own* work.  Copying other people's source code without permission is considered *plagiarism*.
