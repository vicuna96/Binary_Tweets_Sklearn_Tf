import numpy as np
from helper_functions import *
from data_preprocess import *
import argparse
from collections import OrderedDict
from sklearn.svm import SVC

# Adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

clf1 = SVC(gamma='auto', random_state=0)
clf2 = SVC(gamma='auto', random_state=0)
clf3 = SVC(gamma='auto', random_state=0)
clf4 = SVC(gamma='auto', random_state=0)

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


for label, clf, x, y in ensemble_clfs:
    for i in range(30):
        xTr, yTr, xValid, yValid = shuffle(x, y)
        clf.fit(xTr, yTr)

        acc = clf.score(xValid, yValid)
        error_rate[label].append((i, 1-acc))


# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.title('Support Vector machine Estimator')
plt.xlim(0, 30)
plt.xlabel("Run")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
