# K-TSP
This is an implementation of the K-TSP Classifier in Python
The main thing to note is this assumes your data matrix is structered as (# of features, # of samples), like most biological/gene expression datasets
Otherwise this works as most sklearn ML classes. There is a .fit() method that takes in a training data matrix and its respective labels. This finds the optimal k and the k top scoring pairs.
Then a .predict() method that takes in a new training matrix and outputs predictions.

For the training process, there are multiple arguments that can be specified. Here they are listed and some default values are given:
fit(self, data, labels, k_cross_val=True, k=None, verbose=True)
Here "data" is your training matrix and "labels" are the respective labels. "k_cross_val" can be set such that no cross-validation to find k is done, and k is simply 1 if this is false, which is classical TSP. If one already knows how many pairs they would like to use, they can set that as "k", then the "k_cross_val" field is irrelevant. "verbose" is there to print out the optimal value of "k" after the cross validation is done for it. Therefore it only needs to be run once to determine the optimal "k", then the user can pre-define that in subsequent runs.

If there are any bugs encountered when using this code, please raise an issue. Enjoy!
