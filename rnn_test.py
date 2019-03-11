
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn import BasicLSTMCell

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])

#basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
#outputs, states = tf.nn.dynamic_rnn(basic_cell, X, sequence_length=seq_length, dtype=tf.float32)

x = tf.unstack(X, n_steps, 1)

cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)

rnn_outputs = static_bidirectional_rnn(cell_fw,
                                       cell_bw, 
                                       x, 
                                       sequence_length=seq_length,
                                       dtype=tf.float32)
encoder_outputs, encoder_state_fw, encoder_state_bw = rnn_outputs

X_batch = np.array([
  # t = 0      t = 1
  [[0, 1, 2], [9, 8, 7]], # instance 0
  [[3, 4, 5], [0, 0, 0]], # instance 1
  [[6, 7, 8], [6, 5, 4]], # instance 2
  [[9, 0, 1], [3, 2, 1]], # instance 3
])
print(X_batch)
print(X_batch.shape)
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  encoder_outputs_val, encoder_state_fw_val, encoder_state_bw_val = \
    sess.run([encoder_outputs, encoder_state_fw, encoder_state_bw], 
             feed_dict={X: X_batch, seq_length: seq_length_batch})

  print(encoder_outputs_val)
  print()
  print(encoder_state_fw_val)
  print(encoder_state_fw_val[-1])
  print()
  print(encoder_state_bw_val)
  print(encoder_state_bw_val[-1])

"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def BiRNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)
    print(x[0].get_shape())

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        print(batch_x)
        print(batch_x.shape)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        print(batch_x)
        print(batch_x.shape)
        assert 1==0
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
"""