import numpy as np
from helper_functions import *
from data_preprocess import *
import argparse
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier

# Adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

clf1 = KNeighborsClassifier(n_neighbors=3, n_jobs=4)
clf2 = KNeighborsClassifier(n_neighbors=3, n_jobs=4)
clf3 = KNeighborsClassifier(n_neighbors=3, n_jobs=4)
clf4 = KNeighborsClassifier(n_neighbors=3, n_jobs=4)


# Get data using spacy for text
x_train_s, x_test_s, y_train_s = get_data(text_clean)
# Get data tf-idf for text
x_train_f, x_test_f, y_train_f = get_data(normalize_df)
# Get data excluding text
x_train_n, x_test_n, y_train_n = x_train_f.T[:7].T, x_test_f.T[:7].T, y_train_f
# Get data using combined spacy + tf_idf
num_nontext_features = x_train_n.shape[1]
x_train_c = np.hstack((x_train_s, x_train_f[:, num_nontext_features:]))
x_test_c = np.hstack((x_test_s, x_test_f[:, num_nontext_features:]))

ensemble_clfs = [
    ("TF-IDF", clf1, x_train_f, y_train_f),
    ("SPACY", clf2, x_train_s, y_train_s),
    ("NO TEXT", clf3, x_train_n, y_train_n),
    ("COMBINED", clf4, x_train_c, y_train_f)
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _, _, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_neigh = 3
max_neigh = 30

for label, clf, x, y in ensemble_clfs:
    for i in range(min_neigh, max_neigh + 1):
        scores = []
        for _ in range(20):
            clf.set_params(n_neighbors=i)
            xTr, yTr, xValid, yValid = shuffle(x, y, .9)

            clf.fit(xTr, yTr)

            acc = clf.score(xValid, yValid)
            scores.append(1 - acc)
        error_rate[label].append((i, np.mean(scores)))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.title('K-Nearest Neighbors Estimator')
plt.xlim(min_neigh, max_neigh)
plt.xlabel("K Neightbors")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
