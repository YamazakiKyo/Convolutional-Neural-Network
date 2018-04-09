import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsemble
import matplotlib.pyplot as plt

def data_label_split(df, n_labels=0):
    last_col = df.shape[1] - 1
    if n_labels == 2:
        label = df.iloc[:, [last_col-1, last_col]]
    if n_labels == 1:
        label = df.iloc[:, [last_col]]
    data = df.drop(label.columns, axis=1).as_matrix()
    label = label.as_matrix()
    return data, label

def load_data(filename):
    file = pd.read_csv(filename, header=None, skiprows=1)
    # file = pd.get_dummies(file)
    rest, test = train_test_split(file, test_size=0.2, random_state=1)
    train, val = train_test_split(file, test_size=0.25, random_state=1)
    return train, val, test

def onehot(df):
    df_encoded = LabelEncoder().fit_transform(df).reshape(-1, 1)
    df_onehot = OneHotEncoder().fit_transform(df_encoded).toarray()
    return df_onehot

def downsampling(df, n_fold=1):
    last_col = df.shape[1] - 1
    col_index = df.columns
    pos = df[df[col_index[last_col]] == 1] #5K
    neg = df[df[col_index[last_col-1]] == 1] #95K
    balance_ratio = len(pos)/len(neg)
    frame = []
    for _ in range(n_fold):
        neg_frac = neg.sample(frac=balance_ratio)
        down_df = pd.concat([pos, neg_frac], ignore_index=True)
        down_df = shuffle(down_df)
        frame.append(down_df)
    down_df_n = pd.concat(frame)
    return down_df_n

def smotesampling(data, label):
    sm = SMOTEENN()
    data_resampled, label_resampled = sm.fit_sample(data, label)
    return data_resampled, label_resampled

def easyemsemble(data, label, n_subsets=0):
    ee = EasyEnsemble(n_subsets=n_subsets)
    data_resampled, label_resampled = ee.fit_sample(data, label)
    data_resampled = data_resampled.reshape(-1, 12)
    label_resampled = label_resampled.reshape(-1, 1)
    return data_resampled, label_resampled

def construct_layer(input, input_neurons, output_neurons, layer_name, activation=None):
    """An simplified 'tf.layers.Dense' funcion, to clarify what's happening in the fully connected layers"""
    with tf.name_scope(layer_name):
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('Weight'):
            weight = tf.Variable(initializer([input_neurons, output_neurons]))
        with tf.name_scope('bias'):
            bias = tf.Variable(initializer([1, output_neurons]))
        with tf.name_scope('W_dot_x_plus_b'):
            pre_activate = tf.add(tf.matmul(input, weight), bias)
            if activation == None:
                return pre_activate
            else:
                activated = activation(pre_activate)
                return activated


train, val, test = load_data("95k-continuous.csv")
# upsample_train = downsampling(train, n_fold=10)
train_data, train_label = data_label_split(train, n_labels=1)
train_data_res, train_label_res = smotesampling(train_data, train_label)
# train_data_res, train_label_res = easyemsemble(train_data, train_label, n_subsets=10)
train_data_res, train_label_res = shuffle(train_data_res, train_label_res)
train_label_res = onehot(train_label_res)
val_data, val_label = data_label_split(val, n_labels=1)
test_data, test_label = data_label_split(test, n_labels=1)
val_label = onehot(val_label)
test_label = onehot(test_label)

""" Here are some hyperparameters: """
batch_size = 250
iterations = 30000
num_h1 = 30
num_h2 = 20
num_h3 = 10
learning_rate = 0.005
weighted_cost = [1.0, 1.1]


with tf.Graph().as_default() as graph:
    ''' Now let's construct the conputation graph, in TensorFlow, all the compulational processes will follow the graph. '''
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, 12], name='training_data') # each data is flattened from 28x28 = 784
        y_ = tf.placeholder(tf.float32, [None, 2], name='training_label') # after one-hot encoding, each label is a 10-dimention vector

    output_from_H1 = construct_layer(x, 12, num_h1, 'H1', activation=tf.nn.tanh)
    output_from_H2 = construct_layer(output_from_H1, num_h1, num_h2, 'H2', activation=tf.nn.relu)
    output_from_H3 = construct_layer(output_from_H2, num_h2, num_h3, 'H3', activation=tf.nn.relu)

    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(output_from_H3, keep_prob)

    output_layer_logits = construct_layer(dropped, num_h3, 2, 'output', activation=None)
    predicted_label = tf.nn.softmax(output_layer_logits)

    with tf.name_scope('Loss'):
        adjust_weight = tf.constant(weighted_cost)
        adjust_logits = tf.multiply(output_layer_logits, adjust_weight)
        loss = tf.reduce_mean(tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=adjust_logits, labels=y_)))  # cost function = cross entropy
    with tf.name_scope('Train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.6).minimize(loss)
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(predicted_label, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope('Confusion_Matrix'):
        confusion_matrix = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(predicted_label, 1))

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('% Accuracy')
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.set_ylabel('Loss')
    plt.ion()
    plt.show()

    iter_set = np.array([])
    loss_set = np.array([])
    train_acc_set = np.array([])
    val_acc_set = np.array([])

    for i in range(iterations):
        offset = (i * batch_size) % (train_label.shape[0]-batch_size) # make sure the batch will not exceed the data size
        train_data_batch = train_data_res[offset : (offset+batch_size), : ]
        train_label_batch = train_label_res[offset : (offset + batch_size), :]
        feed_dict_train = {
            x : train_data_batch,
            y_ : train_label_batch,
            keep_prob: 0.9
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
        _, los, acc = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict_train)
        if (i % 50 == 0):
            acc2 = sess.run(accuracy, feed_dict=feed_dict_val)
            iter_set = np.append(iter_set, i)
            loss_set = np.append(loss_set, los)
            train_acc_set = np.append(train_acc_set, acc)
            val_acc_set = np.append(val_acc_set, acc2)
            try:
                ax.lines.remove(line1)
                ax.lines.remove(line2)
                ax2.lines.remove(line3)
            except Exception:
                pass
            line1, = ax.plot(iter_set, train_acc_set, '-r', label='Training_acc')
            line2, = ax.plot(iter_set, val_acc_set, '-b', label='Validation_acc')
            line3, = ax2.plot(iter_set, loss_set, '-k', label = 'loss')
            ax.legend(loc=3)
            ax2.legend(loc=8)
            plt.pause(0.1)
            print('Validation accuracy at iteration %s: %s' %(i, acc2))
    acc3, confMaX = sess.run([accuracy, confusion_matrix], feed_dict=feed_dict_test)
    print("Test accuracy is: %s" %acc3)
    print("The test confusion matrix is ")
    print(confMaX)
