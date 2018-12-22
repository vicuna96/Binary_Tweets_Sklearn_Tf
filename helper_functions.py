import time
import datetime
import numpy as np
from string import punctuation
import matplotlib.pyplot as plt

count_http = lambda x: x.count("http")

nan_to_0 = lambda x: 0 if np.isnan(x) else 1

bool_to_int = lambda x: 1 if x else 0

remove_ats = lambda x: remove_sym(x, "@")

remove_http = lambda x: remove_sym(x, "http")


def remove_sym(x, symbol):
    ind_start = x.find(symbol)
    while ind_start != -1:
        ind_end = x.find(" ", ind_start)
        if ind_end != -1:
            x = x[:ind_start]+x[ind_end:]
        else:
            x = x[:ind_start]
        ind_start = x.find(symbol)
    x = x.strip()
    return x


def get_ats(x):
    lst = []
    for _ in range(x.count("@")):
        ind_start = x.find("@")
        ind_end = x.find(" ", ind_start)
        lst.append(x[ind_start:ind_end])
        if ind_end != -1:
            x = x[:ind_start]+x[ind_end:]
        else:
            x = x[:ind_start]
        x = x.strip()
    return lst


def to_unix(x):
    try:
        return time.mktime(datetime.datetime.strptime(x, "%m/%d/%y").timetuple())
    except:
        return time.mktime(datetime.datetime.strptime(x, "%m/%d/%Y").timetuple())


def to_seconds(string):
    ind = string.find(' ')
    time = string[ind+1:]
    lst = time.split(':')
    seconds = int(int(lst[0])*60+int(lst[1]))
    return seconds


def to_days(string):
    ind = string.find(' ')
    date = string[:ind]
    days = int(to_unix(date)//86400) - int(to_unix('12/14/15')//86400)
    return days


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


def shuffle(xTr, yTr, split=.8):
    n, d = xTr.shape

    indices = np.random.permutation(n)
    index = int(np.ceil(split*n))

    train, validate = indices[:index], indices[index:]

    xValid, yValid = xTr[validate, :], yTr[validate]
    xTr, yTr = xTr[train, :], yTr[train]

    return xTr, yTr, xValid, yValid


def shuffle_ensemble(data, split=.8):
    n = data[0][0].shape[0]

    indices = np.random.permutation(n)
    index = int(np.ceil(split*n))

    train, validate = indices[:index], indices[index:]

    dataTrain = [(xTr[train, :], yTr[train]) for xTr, yTr in data]
    xValid = [xTr[validate, :] for xTr, _ in data]
    yValid = [yTr[validate] for _, yTr in data]

    return dataTrain, xValid, yValid


def write_pred(pred, file="submission.csv"):
    f = open(file, "w")
    f.write("ID,Label\n")
    for i in range(len(pred)):
        f.write(str(i)+','+str(pred[i])+'\n')


def plot_hist(hist):
    plt.figure(1)

    plt.subplot(411)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(412)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()
    return


def bagging_keras(xTr, yTr, xTe, models, epochs, batch=128, plot=False):
    n, d = xTr.shape
    predictions = []
    valid_accuracies = []

    indices = np.arange(n)
    for model in models:
        sub = np.random.choice(indices, n)
        val_indices = np.array([i for i in range(n) if i not in set(sub)])
        hist = model.fit(xTr[sub], yTr[sub], verbose=2, epochs=epochs, batch_size=batch)
        validation_accu = np.mean(model.predict_classes(xTr[val_indices]) != yTr[val_indices])
        valid_accuracies.append(validation_accu)
        print("Final validation accuracy is " + str(validation_accu))
        pred = model.predict_classes(xTe).flatten()
        pred[pred == 0] = -1
        predictions.append(pred)
        if plot:
            plot_hist(hist)

    return np.array(predictions), models, valid_accuracies


def train_ten(xTr, yTr, xTe, models, epochs, batch=128):
    n, d = xTr.shape
    predictions = []
    valid_accuracies = []

    indices = np.arange(n)
    for model in models:
        indices = np.random.permutation(indices)
        hist = model.fit(xTr[indices], yTr[indices], validation_split=0.2, verbose=2, epochs=epochs, batch_size=batch)
        # validation_accu = np.mean(model.predict_classes(xTr[val_indices]) != yTr[val_indices])
        valid_accuracies.append(hist.history['val_acc'][-1])
        print("Final validation accuracy is " + str(hist.history['val_acc'][-1]))
        pred = model.predict_classes(xTe).flatten()
        pred[pred == 0] = -1
        predictions.append(pred)
        # plot_hist(hist)

    return np.array(predictions), models, valid_accuracies
