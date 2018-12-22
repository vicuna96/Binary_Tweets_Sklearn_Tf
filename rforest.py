import numpy as np
from helper_functions import *
from data_preprocess import *
import argparse
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier

# Adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

clf1 = RandomForestClassifier(n_estimators=100, random_state=4,verbose=0,oob_score=True,warm_start=True)
clf2 = RandomForestClassifier(n_estimators=100, random_state=4,verbose=0,oob_score=True,warm_start=True)
clf3 = RandomForestClassifier(n_estimators=100, random_state=4,verbose=0,oob_score=True,warm_start=True)
clf4 = RandomForestClassifier(n_estimators=100, random_state=4,verbose=0,oob_score=True,warm_start=True)

# Get data using spacy for text
x_train_s, x_test_s, y_train_s = get_data(text_clean)
# Get data tf-idf for text
x_train_f, x_test_f, y_train_f = get_data(normalize_df)
# Get data excluding text
x_train_n, x_test_n, y_train_n = x_train_f.T[:7].T, x_test_f.T[:7].T, y_train_f
# Get data using combined spacy + tf_idf
num_nontext_features = x_train_n.shape[1]
x_train_c = np.hstack((x_train_s,x_train_f[:,num_nontext_features:]))
x_test_c = np.hstack((x_test_s,x_test_f[:,num_nontext_features:]))

ensemble_clfs = [
    ("TF-IDF", clf1, x_train_f, y_train_f),
    ("SPACY", clf2, x_train_s, y_train_s),
    ("NO TEXT", clf3, x_train_n, y_train_n),
    ("COMBINED", clf4, x_train_c, y_train_f)
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _, _, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 300

for label, clf, x, y in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(x, y)

        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.title('Random Forest Estimator')
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
