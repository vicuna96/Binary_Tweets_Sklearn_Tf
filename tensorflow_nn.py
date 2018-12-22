from data_preprocess import *
from helper_functions import write_pred
import tensorflow as tf
import argparse

# Define the parser to run script on cmd
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-epochs", action='store', type=int, required=True,
                    help="Number of epochs to train for")
parser.add_argument("-dir", action='store', type=str, required=False,
                    help="Directory for the logs")
parser.add_argument("-batch", action='store', type=int, required=False,
                    help="Batch size to train with")
parser.add_argument("-lr", action='store', type=float, required=False,
                    help="Learning rate")
args = parser.parse_args()

print(parser.parse_args())

epochs = args.epochs
if args.batch:
    batch_size = args.batch
else:
    batch_size = 64
if args.dir:
    summaries_dir = args.dir
else:
    summaries_dir = "C:\\Users\\danis_000\\Desktop\\EnsembleNet\\logdir"
if args.lr:
    learning_rate = args.lr
else:
    learning_rate = .001
print(summaries_dir)

# tf.set_random_seed(1234)

# Get data tf-idf for text
x_train_f, x_test_f, y_train_f = get_data(normalize_df)
x_train_f, x_test_f, y_train_f = x_train_f.astype(np.float32), x_test_f.astype(np.float32), y_train_f.astype(np.float32)

split_ind = int(.7*x_train_f.shape[0])
# xTr, yTr, xValid, yValid = shuffle(x_train_f,y_train_f)
xTr, xValid = x_train_f[:split_ind], x_train_f[split_ind:]
yTr, yValid = y_train_f[:split_ind], y_train_f[split_ind:]

# Probability of dropout
prob = tf.placeholder_with_default(1.0, shape=())
# Place holder for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, x_train_f.shape[1]], name="Inputs")
outputs = tf.placeholder(tf.float32, [None, 2], name="Outputs")

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([x_train_f.shape[1], 32], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([32]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([32, 12], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([12]), name='b2')
# and the weights connecting the hidden layer to the output layer
W3 = tf.Variable(tf.random_normal([12, 2], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([2]), name='b3')

# calculate the output of the hidden layer
hidden = tf.add(tf.matmul(inputs, W1), b1)
hidden = tf.nn.relu(hidden)
hidden = tf.layers.dropout(hidden, prob)
# calculate the output of the hidden layer
hidden = tf.add(tf.matmul(hidden, W2), b2)
hidden = tf.nn.relu(hidden)
hidden = tf.layers.dropout(hidden, prob)
# calculate the inputs of the second hidden layer
hidden = tf.add(tf.matmul(hidden, W3), b3)
outputs_ = tf.nn.softmax(hidden)

print("Shape inputs is "+str(inputs.shape))
print("Shape labels is "+str(outputs.shape))
print("Shape outputs_ is "+str(outputs_.shape))


cross_entropy = tf.reduce_mean(-tf.reduce_sum(outputs * tf.log(outputs_ + 1e-10)
                                              + (1.0 - outputs) * tf.log(1.0 - outputs_ + 1e-10), axis=1))

# Set params
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(outputs_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def to_tensors(xtrain, ytrain):
    x_tensor = tf.convert_to_tensor(xtrain)
    y_tensor = tf.convert_to_tensor(ytrain)
    y_tensor = tf.cast(y_tensor, tf.int32)
    y_tensor = tf.one_hot(y_tensor, 2)

    return x_tensor, y_tensor


x_train_tensor, y_train_tensor = to_tensors(xTr, yTr)
x_valid_tensor, y_valid_tensor = to_tensors(xValid, yValid)


trainset = tf.data.Dataset.from_tensor_slices((x_train_tensor, y_train_tensor))
trainset = trainset.shuffle(batch_size).batch(batch_size)

iterator = trainset.make_initializable_iterator()
# extract an element
next_element = iterator.get_next()
# initialize a session
sess = tf.Session()
# add cross entropy to logs
with tf.name_scope('cross_entropies'):
    cross_entropies = cross_entropy
tf.summary.scalar('cross_entropy', cross_entropies)
# add out-of-box accuracy to logs
with tf.name_scope('accuracies'):
    accuracies = accuracy
tf.summary.scalar('accuracy', accuracies)
merged = tf.summary.merge_all()
# set the directory used for logs
train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

# compute number of batches
batches = xTr.shape[0] // batch_size
# initialize variables
sess.run(init_op)
print(tf.trainable_variables())
# initialize iterator
sess.run(iterator.initializer)
# initialize training
for epoch in range(epochs):
    avg_cost = 0
    for i in range(batches):
        batch_x, batch_y = sess.run(next_element)
        summary, _, ent, acc = sess.run([merged, optimiser, cross_entropy, accuracy],
                                        feed_dict={inputs: batch_x, outputs: batch_y, prob: 0.6})
        avg_cost += ent / batches
        train_writer.add_summary(summary, epoch*batches + i)
    dict = {inputs: x_valid_tensor.eval(session=sess), outputs: y_valid_tensor.eval(session=sess)}
    oob_acc = sess.run(accuracy, feed_dict=dict)
    print("Out-of-Box accuracy is " + str(oob_acc))
    # Stop if there is symptoms of over-fitting
    if acc > oob_acc > .9:
        break
    sess.run(iterator.initializer)
    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "acc =", "{:.3f}".format(acc))

# get predictions using same graph
pred = outputs_.eval(feed_dict={inputs: x_test_f}, session=sess)
pred = np.argmax(pred, axis=1)
# Change predictions from {0,1} to {-1,1}
pred[pred == 0] = -1
print("Sum of labels is " + str(np.sum(pred)))

# write predictions to default file
write_pred(pred)
