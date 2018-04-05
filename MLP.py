import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_data():
    file = pd.read_csv("MNIST.csv")
    label = file[['label']].values
    one_hot = OneHotEncoder() # One-hot encoding the label
    one_hot.fit(label)
    label = one_hot.transform(label).toarray()
    data = file.drop(['label'], axis=1).values
    rest, test_data, rest_label, test_label = train_test_split(data, label, test_size=0.2, random_state=1) # Split 20% as test data
    train_data, val_data, train_label, val_label = train_test_split(rest, rest_label , test_size=0.25, random_state=1)
    return train_data, train_label, val_data, val_label, test_data, test_label

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def construct_layer(input, input_neurons, output_neurons, layer_name, activation=None):
    """An simplified 'tf.layers.Dense' funcion, to clarify what's happening in the fully connected layers"""
    with tf.name_scope(layer_name):
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('Weight'):
            weight = tf.Variable(initializer([input_neurons, output_neurons]))
            variable_summaries(weight)
        with tf.name_scope('bias'):
            bias = tf.Variable(initializer([1, output_neurons]))
            variable_summaries(bias)
        with tf.name_scope('W_dot_x_plus_b'):
            pre_activate = tf.add(tf.matmul(input, weight), bias)
            tf.summary.histogram('pre_activations', pre_activate)
            if activation == None:
                return pre_activate
            else:
                activated = activation(pre_activate)
                tf.summary.histogram('activations', activated)
                return activated

train_data, train_label, val_data, val_label, test_data, test_label = load_data()
# train : validation : test = 6 : 2 ：2
""" Here are some hyper-parameters: """
batch_size = 100 # Here we applied the mini-batch gradient decsend
epoches = 30000

with tf.Graph().as_default() as graph:
    ''' Now let's construct the conputation graph, in TensorFlow, all the compulational processes will follow the graph. '''
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, 784], name='training_data') # each data is flattened from 28x28 = 784
        y_ = tf.placeholder(tf.float32, [None, 10], name='training_label') # after one-hot encoding, each label is a 10-dimention vector

    output_from_H1 = construct_layer(x, 784, 777, 'H1', activation=tf.nn.tanh)
    output_from_H2 = construct_layer(output_from_H1, 777, 666, 'H2', activation=tf.nn.relu)
    output_from_H3 = construct_layer(output_from_H2, 666, 555, 'H3', activation=tf.nn.relu)

    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(output_from_H3, keep_prob)

    output_layer_logits = construct_layer(dropped, 555, 10, 'output', activation=None)
    predicted_label = tf.nn.softmax(output_layer_logits)

    # Officially, here is a easier way, 3-layer NN as an example:
    # output_from_H1 = tf.layers.dense(inputs=tf_train_data, units=777, activation=tf.nn.relu)
    # output_from_H2 = tf.layers.dense(inputs=output_from_H1, units=666, activation=tf.nn.relu)
    # output_layer_logits = tf.layers.dense(inputs=output_from_H2, units=10, activation=None)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer_logits, labels=y_))  # cost function = cross entropy
        tf.summary.scalar('Loss', loss)
    with tf.name_scope('Train'):
        optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(predicted_label, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

with tf.Session(graph=graph) as sess:
    merge_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("log/train", sess.graph)
    val_writer = tf.summary.FileWriter("log/validation", sess.graph)
    test_writer = tf.summary.FileWriter("log/test", sess.graph)
    tf.global_variables_initializer().run()

    for i in range(epoches):
        offset = (i * batch_size) % (train_label.shape[0]-batch_size) # make sure the batch will not exceed the data size
        train_data_batch = train_data[offset : (offset+batch_size), : ]
        train_label_batch = train_label[offset : (offset + batch_size), :]
        feed_dict_train = {
            x : train_data_batch,
            y_ : train_label_batch,
            keep_prob: 0.1
        }
        feed_dict_val = {
            x: val_data,
            y_: val_label,
            keep_prob: 1.0
        }
        feed_dict_test = {
            x: test_data,
            y_: test_label,
            keep_prob: 1.0
        }
        if (i % 10 ==0):
            summary, acc = sess.run([merge_summary, accuracy], feed_dict=feed_dict_val)
            val_writer.add_summary(summary, i)
            print('Validation accuracy at epoch %s: %s' %(i, acc))
            if (acc > 0.95):
                break
        else:
            summary, _ = sess.run([merge_summary, optimizer], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)
    summary, acc2 = sess.run([merge_summary, accuracy], feed_dict=feed_dict_test)
    test_writer.add_summary(summary)
    print("Test accuracy is: %s" %acc2)

