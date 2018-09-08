# K-nearest neighbour
# Goals
KNN is a popular machine learning algorithm and (as far as I know at the time of this writing) a good classifier for linearly solvable,
low dimensional problems.
<br>
Also it is great to consolidate several of the foundations for machine learning and I get to experiment a bit more with python and
libraries that will also be needed in more complex examples.
# General information
After finishing the network I compared it to scikit's KNeighborsClassifier and while my network works, it performs rather poorly.
<br>
I used the iris dataset with a ratio of 70/20/10 for training, validation and test dataset and even though my network had the exact same neighbours (for k=5) as scikit's network, mine had only 4 correct guesses, while scikit was at 19.
<br>
Since the neighbours were the same, I suspect that there are problems with edge cases, e.g. there is a tie between two classes, where you could compare the total distances to resolve the tie.
<br>
As this isn't that complicated but tedious to resolve, I decided against it for now, in case I stumble upon this problem in the future again I might resolve it then.
