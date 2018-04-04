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

def construct_layer(input, input_neurons, output_neurons, nth_layer, activation = None):
    current_layer = 'Layer%s' % nth_layer
    with tf.name_scope(current_layer):
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('Weight'):
            weight = tf.Variable(initializer([input_neurons, output_neurons]))
            tf.summary.histogram(current_layer + '/weight', weight)
        with tf.name_scope('bias'):
            bias = tf.Variable(initializer([1, output_neurons]))
            tf.summary.histogram(current_layer + '/bias', bias)
        with tf.name_scope('W_dot_x_plus_b'):
            output = tf.add(tf.matmul(input, weight), bias)
            if activation == None:
                return output
            else:
                return activation(output)

def cal_accuracy(predicted, desired):
    return (100.0 * np.sum(np.argmax(predicted, 1) == np.argmax(desired, 1)) / predictions.shape[0])
    # correct_predicts = tf.equal(tf.argmax(predicted, 1), tf.argmax(desired, 1))
    # bool_to_float = tf.cast(correct_predicts, tf.float32)
    # accuracy = tf.reduce_mean(bool_to_float)
    # return accuracy

train_data, train_label, val_data, val_label, test_data, test_label = load_data()
# train : validation : test = 6 : 2 ：2

""" Here are some hyper-parameters: """
batch_size = 100 # Here we applied the mini-batch gradient decsend
epoches = 300

with tf.Graph().as_default() as graph:
    """ Now let's construct the conputation graph, in TensorFlow, all the compulational processes will follow the graph. """
    with tf.name_scope('Train'):
        tf_train_data = tf.placeholder(tf.float32, [batch_size, 784], name='training_data') # each data is flattened from 28x28 = 784
        tf_train_label = tf.placeholder(tf.float32, [batch_size, 10], name='training_label') # after one-hot encoding, each label is a 10-dimention vector
    with tf.name_scope('Validation'):
        tf_val_data = tf.constant(val_data, dtype=tf.float32, name='validation_data')
    with tf.name_scope('Test'):
        tf_test_data = tf.constant(test_data, dtype=tf.float32, name='test_data')

    with tf.variable_scope('Input_to_H1') as scope:
        output_from_H1 = construct_layer(tf_train_data, 784, 777, nth_layer=1, activation=tf.nn.tanh)
        scope.reuse_variables()
        val_from_H1 = construct_layer(tf_val_data, 784, 777, nth_layer=1, activation=tf.nn.tanh)
        test_from_H1 = construct_layer(tf_test_data, 784, 777, nth_layer=1, activation=tf.nn.tanh)
    with tf.variable_scope('H1_to_H2') as scope:
        output_from_H2 = construct_layer(output_from_H1, 777, 666, nth_layer=2, activation=tf.nn.relu)
        scope.reuse_variables()
        val_from_H2 = construct_layer(val_from_H1, 777, 666, nth_layer=2, activation=tf.nn.relu)
        test_from_H2 = construct_layer(test_from_H1, 777, 666, nth_layer=2, activation=tf.nn.relu)
    with tf.variable_scope('H2_to_H3') as scope:
        output_from_H3 = construct_layer(output_from_H2, 666, 555, nth_layer=3, activation=tf.nn.relu)
        scope.reuse_variables()
        val_from_H3 = construct_layer(val_from_H2, 666, 555, nth_layer=3, activation=tf.nn.relu)
        test_from_H3 = construct_layer(test_from_H2, 666, 555, nth_layer=3, activation=tf.nn.relu)
    with tf.variable_scope('H3_to_Output') as scope:
        output_layer_logits = construct_layer(output_from_H3, 555, 10, nth_layer=4, activation=None)
        predicted_label = tf.nn.softmax(output_layer_logits)
        scope.reuse_variables()
        val_from_output = construct_layer(val_from_H3, 555, 10, nth_layer=4, activation=tf.nn.softmax)
        test_from_output = construct_layer(test_from_H3, 555, 10, nth_layer=4, activation=tf.nn.softmax)

    # output_from_H1 = tf.layers.dense(inputs=tf_train_data, units=777, activation=tf.nn.relu)
    # output_from_H2 = tf.layers.dense(inputs=output_from_H1, units=666, activation=tf.nn.relu)
    # output_layer_logits = tf.layers.dense(inputs=output_from_H2, units=10, activation=None)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer_logits, labels=tf_train_label))  # cost function = cross entropy
        tf.summary.scalar('Loss', loss)
    with tf.name_scope('Optimization'):
        optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

with tf.Session(graph=graph) as sess:
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("log/", sess.graph)
    tf.global_variables_initializer().run()
    for i in range(epoches):
        offset = (i * batch_size) % (train_label.shape[0]-batch_size) # make sure the batch will not exceed the data size
        train_data_batch = train_data[offset : (offset+batch_size), : ]
        train_label_batch = train_label[offset : (offset + batch_size), :]
        feed_dict = {
            tf_train_data : train_data_batch,
            tf_train_label : train_label_batch
        }
        _, batch_loss, predictions = sess.run([optimizer, loss, predicted_label], feed_dict=feed_dict)
        if (i % 100 == 0):
            print("Minibatch loss at step %d: %f" % (i, batch_loss))
            print("Minibatch accuracy: %.1f%%" % cal_accuracy(predictions, train_label_batch))
            print("Validation accuracy: %.1f%%" % cal_accuracy(val_from_output.eval(), val_label))
    print("Test accuracy: %.1f%%" % cal_accuracy(test_from_output.eval(), test_label))


















