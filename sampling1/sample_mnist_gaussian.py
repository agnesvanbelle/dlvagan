import pandas as pd
import os
from scipy.stats import multivariate_normal, rv_discrete
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

class Sampler():

  def fit(self, X, Y):
    label_and_count = list(zip(*[list(x) for x in np.unique(Y, return_counts=True)]))
    self.label_to_prob = {k: v / float(len(Y)) for k, v in label_and_count}

    self.models = {}
    labels = set(Y)
    for label in labels:
      m = self._make_gaussian_model(label, X, Y)
      self.models[label] = m

  def _make_gaussian_model(self, label, X, Y):
    indices = np.where(Y == label)
    my_mean = np.mean(X[indices], axis=0)
    my_cov = np.cov(X[indices].T)
    return {'mean': my_mean, 'cov': my_cov}

  def sample_given_y(self, y):
    m = self.models[y]
    return multivariate_normal.rvs(mean=m['mean'], cov=m['cov'])

  def sample(self):
    label_to_prob_items = sorted(self.label_to_prob.items())
    label_distr = rv_discrete(values=([x[0] for x in label_to_prob_items], [x[1] for x in label_to_prob_items]))
    sampled_y = label_distr.rvs()
    return sampled_y, self.sample_given_y(sampled_y)

def visualize_mnist_x(label, pixels):
  pixels = pixels.reshape((28, 28))
  plt.title('Label is {label}'.format(label=label))
  plt.imshow(pixels, cmap='gray')
  plt.show()

mnist_file = os.path.join(os.path.dirname(__file__), '../large_files/train.csv')
df = pd.read_csv(mnist_file)
(n_rows, n_columns) = df.shape

s = Sampler()
X = df.iloc[:,1:].to_numpy(dtype=float)
Y = df.iloc[:,0].to_numpy(dtype=int)
s.fit(X, Y)

visualize_mnist_x(1, s.sample_given_y(1))
for i in range(5):
  sampled_label, sample = s.sample()
  visualize_mnist_x(sampled_label, sample)

