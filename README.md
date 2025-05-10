# K-TSP
This is an implementation of the K-TSP Classifier in Python
The main thing to note is this assumes your data matrix is structered as (# of features, # of samples), like most biological/gene expression datasets
Otherwise this works as most sklearn ML classes. There is a .fit method that takes in a training data matrix and its respective labels. This finds the optimal k and the k top scoring pairs.
Then a .predict method that takes in a new training matrix and outputs predictions.
