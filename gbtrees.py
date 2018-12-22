import numpy as np
from helper_functions import *
from data_preprocess import *
from sklearn.ensemble import GradientBoostingClassifier
from collections import OrderedDict

# Adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

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


clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=4, random_state=0, warm_start=False)
clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=4, random_state=0, warm_start=False)
clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=4, random_state=0, warm_start=False)
clf4 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=4, random_state=0, warm_start=False)

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
max_estimators = 50

for label, clf, x, y in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        scores = []
        clf.set_params(n_estimators=i)
        for _ in range(4):
            xTr, yTr, xValid, yValid = shuffle(x, y)
            xTr_c, yTr_c = np.copy(xTr), np.copy(yTr)
            xValid_c, yValid_c = np.copy(xValid), np.copy(yValid)
            clf.fit(xTr_c, yTr_c)

            acc = clf.score(xValid_c, yValid_c)
            scores.append(1 - acc)
        error_rate[label].append((i, np.mean(scores)))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.title('Gradient Boosted Trees Estimator')
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
