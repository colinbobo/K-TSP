import pandas as pd
import numpy as np
from copy import deepcopy

# Algorithm for k-TSP
class KTSP:

  def __init__(self):
    self.data = None
    self.labels = None
    self.rank_matrix = None
    self.best_pair = None
    self.top_k_pairs = None

  def compute_Rank(self, reduced_data : pd.DataFrame):
    """
    for i in range(reduced_data.shape[1]):
      sorted_col = reduced_data[i].sort_values()
      rank = 1
      for num in sorted_col:
        reduced_data[i].loc[num] = rank
        rank += 1
    """
    # faster way based on built-in pandas method
    self.rank_matrix = reduced_data.rank(method="first", ascending=False, axis=0)  # Compute ranks per column
    return self.rank_matrix.astype(int)


  def compute_Delta(self):
    # initializing the delta matrix, it's P X P symmetric
    delta =  np.zeros((self.P,self.P))

    # we subset the matrix based on classes
    class_0 = self.rank_matrix.loc[:, self.labels == 0]
    class_1 = self.rank_matrix.loc[:, self.labels == 1]

    for i in range(self.P):
      for j in range(i+1,self.P):
        p_class0 = (class_0.iloc[i] < class_0.iloc[j]).mean()
        p_class1 = (class_1.iloc[i] < class_1.iloc[j]).mean()
        delta[i, j] = abs(p_class0 - p_class1)
        delta[j, i] = delta[i, j]

    return pd.DataFrame(delta)

  def compute_Gamma(self):
    # initializing the gamma matrix which is also P X P
    gamma_mat = np.zeros((self.P,self.P))

    # check that this gets all columns with the corresponding label
    class_0 = self.rank_matrix.loc[:, self.labels == 0].to_numpy()
    class_1 = self.rank_matrix.loc[:, self.labels == 1].to_numpy()

    for i in range(self.P):
      for j in range(i+1, self.P):
        #check this
        gamma_0 = np.mean(class_0[i, :] - class_0[j, :])
        gamma_1 = np.mean(class_1[i, :] - class_1[j, :])
        gamma_mat[i, j] = abs(gamma_0 - gamma_1)
        gamma_mat[j, i] = gamma_mat[i, j]

    return pd.DataFrame(gamma_mat)


  def get_ordered_list(self, delta_mat, gamma_mat):
    # stack Delta and Gamma to sort pairs by Delta, then Gamma
    delta_stacked = delta_mat.stack()
    gamma_stacked = gamma_mat.stack()
    pairs_df = pd.DataFrame({
        'delta': delta_stacked,
        'gamma': gamma_stacked
    })
    # sort by Delta, then Gamma
    pairs_df = pairs_df.sort_values(by=['delta', 'gamma'], ascending=[False, False])
    # select top k pairs
    top_pairs = [(i, j) for i, j in pairs_df.index if i < j]
    return top_pairs


  def remove_pair(self, top_pair, list_O):
    return [(i,j) for (i,j) in list_O if i not in top_pair and j not in top_pair]


  def cross_val_k(self):
    k_upper_bound = 10 # should be even, since it'll exclude the last number in the loop below
    N = self.labels.shape[0]
    little_n = 3
    m = N // little_n
    sample_indices = np.arange(self.data.shape[1])
    np.random.shuffle(sample_indices)
    error_rates = {}

    for fold in range(m):
      test_indices = sample_indices[fold*little_n : (fold+1)*little_n]
      train_indices = np.setdiff1d(sample_indices, test_indices)

      X_train = self.data.iloc[:, train_indices]
      X_test = self.data.iloc[:, test_indices]
      y_train = self.true_labels.iloc[train_indices]
      y_test = self.true_labels.iloc[test_indices]

      self.labels = pd.Series(y_train.values, index=X_train.columns)
      self.P = X_train.shape[0]
      self.compute_Rank(X_train)
      delta_mat = self.compute_Delta()
      gamma_mat = self.compute_Gamma()

      list_O = self.get_ordered_list(delta_mat, gamma_mat)
      list_Theta = []
      for k in range(1,k_upper_bound):
        list_Theta.append(list_O[0])
        list_O = self.remove_pair(list_O[0], list_O)
        if k % 2 != 0:
          self.top_k_pairs = list_Theta
          self.optimal_k = k
          if self.optimal_k == 1:
            self.best_pair = self.top_k_pairs[0]
          predictions = self.predict(X_test)
          accuracy = (predictions == y_test.values).mean()
          # print(f"Accuracy for k={k} is: {accuracy}")
          error_rate = 1 - accuracy
          error_rates[k] = error_rate

    self.optimal_k = min(error_rates, key=error_rates.get)
    self.labels = pd.Series(self.true_labels.values, index=self.data.columns)
    self.P = self.data.shape[0]
    self.compute_Rank(self.data)
    delta_mat = self.compute_Delta()
    gamma_mat = self.compute_Gamma()
    ordered_list = self.get_ordered_list(delta_mat, gamma_mat)
    self.top_k_pairs = []
    for _ in range(self.optimal_k):
      self.top_k_pairs.append(ordered_list[0])
      ordered_list = self.remove_pair(ordered_list[0], ordered_list)
    return self.optimal_k

  def fit(self, data, labels, k_cross_val=True, k=None, verbose=False):
    self.true_labels = deepcopy(labels)
    self.k_cross_val = k_cross_val
    if (self.k_cross_val == False and (k == None or k == 1)) or k == 1:
      self.optimal_k = 1
      self.data = data
      self.labels = labels
      # this makes sure that we have the correct orientation for the data
      if len(self.labels) == self.data.shape[0]:
          self.data = self.data.T
      elif len(self.labels) != self.data.shape[1]:
          raise Exception(f"Number of samples in data ({self.data.shape[1]}) must match length of labels ({len(self.labels)})")

      self.labels = pd.Series(self.labels.values, index=self.data.columns)
      self.P = self.data.shape[0]
      self.compute_Rank(self.data)
      delta_mat = self.compute_Delta()
      gamma_mat = self.compute_Gamma()

      delta_arr = delta_mat.to_numpy()
      max_delta = delta_arr.max()
      max_indices = np.where(delta_arr == max_delta)
      max_pairs = list(zip(max_indices[0], max_indices[1]))

      if len(max_pairs) == 1:
        return max_pairs[0]
      else:
        max_gamma = -float('inf')
        self.best_pair = None
        for i, j in max_pairs:
            gamma_value = gamma_mat.iloc[i, j]
            if gamma_value > max_gamma:
                max_gamma = gamma_value
                self.best_pair = (i, j)
        return self.best_pair
    else:
      self.data = data
      self.labels = labels
      if k == None:
        self.cross_val_k()
        if verbose:
          print(f"Optimal k: {self.optimal_k}")
      else:
        self.optimal_k = k
        self.labels = pd.Series(self.true_labels.values, index=self.data.columns)
        self.P = self.data.shape[0]
        self.compute_Rank(self.data)
        delta_mat = self.compute_Delta()
        gamma_mat = self.compute_Gamma()
        ordered_list = self.get_ordered_list(delta_mat, gamma_mat)
        self.top_k_pairs = []
        for _ in range(self.optimal_k):
          self.top_k_pairs.append(ordered_list[0])
          ordered_list = self.remove_pair(ordered_list[0], ordered_list)


  def predict(self, new_samples):
    if isinstance(new_samples, pd.Series):
        new_samples = new_samples.to_frame().T
    if (self.k_cross_val == False and (self.optimal_k == None or self.optimal_k == 1)) or self.optimal_k == 1:
      i = self.best_pair[0]
      j = self.best_pair[1]

      # compute p_ij(C_m)
      class_0 = self.rank_matrix.loc[:, self.labels == 0]
      class_1 = self.rank_matrix.loc[:, self.labels == 1]
      p_class0 = (class_0.iloc[i] < class_0.iloc[j]).mean()
      p_class1 = (class_1.iloc[i] < class_1.iloc[j]).mean()

      # decision rule
      predictions = []
      for _, sample in new_samples.items():
        new_sample = pd.Series(sample, index=self.data.index)
        new_ranks = pd.Series(new_sample).rank(method="first", ascending=False)
        votes = {0: 0, 1: 0}
        if p_class0 > p_class1:
          if new_ranks.iloc[i] < new_ranks.iloc[j]:
            votes[0] += 1
          else:
            votes[1] += 1
        else:
          if new_ranks.iloc[i] < new_ranks.iloc[j]:
            votes[1] += 1
          else:
            votes[0] += 1
        predictions.append(0 if votes[0] > votes[1] else 1)
      return np.array(predictions)
    else:
      if self.top_k_pairs is None:
        raise Exception("Must call fit() to select top k pairs before making predictions")

      predictions = []
      for _, sample in new_samples.items():
        new_sample = pd.Series(sample, index=self.data.index)
        new_ranks = pd.Series(new_sample).rank(method="first", ascending=False)
        votes = {0: 0, 1: 0}
        for i, j in self.top_k_pairs:
          class_0 = self.rank_matrix.loc[:, self.labels == 0]
          class_1 = self.rank_matrix.loc[:, self.labels == 1]
          p_class0 = (class_0.iloc[i] < class_0.iloc[j]).mean()
          p_class1 = (class_1.iloc[i] < class_1.iloc[j]).mean()
          if p_class0 > p_class1:
            if new_ranks.iloc[i] < new_ranks.iloc[j]:
                votes[0] += 1
            else:
                votes[1] += 1
          else:
            if new_ranks.iloc[i] < new_ranks.iloc[j]:
                votes[1] += 1
            else:
                votes[0] += 1
        predictions.append(0 if votes[0] > votes[1] else 1)
      return np.array(predictions)

    