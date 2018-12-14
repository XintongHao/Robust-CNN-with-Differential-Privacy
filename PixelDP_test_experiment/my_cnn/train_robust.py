import mnist_input
# import mnist_model
# reload(mnist_model)
import numpy as np
import tensorflow as tf
mnist = mnist_input.read_data_sets('MNIST_data', one_hot=True)

import mnist_noise_model
from importlib import reload

reload(mnist_noise_model)

# checkpoint_path = "save_models/baseline_40epochs.ckpt"
checkpoint_path = "save_models/robust_model/"
x, y_ = mnist_noise_model.place_holders()
y_conv, keep_prob, variable_dict = mnist_noise_model.model(x)

################# Sess ##############################


cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))) #avoid 0*log(0) error

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# saver = tf.train.Saver(variable_dict)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

absolute_sums = []
for variable in variable_dict.values():
    absolute_sums.append(tf.reduce_sum(tf.abs(variable)))

################# Validation Accuracy ##############################

def print_test_accuracy(test_acc_list):
    idx = 0
    batch_size = 500
    num_correct = 0
    while(idx < len(mnist.test.images)):
        num_correct += np.sum(correct_prediction.eval(feed_dict = {
               x: mnist.test.images[idx:idx+batch_size], 
               y_: mnist.test.labels[idx:idx+batch_size], keep_prob: 1.0
                    }))
        idx+=batch_size
    test_acc = float(num_correct)/float(len(mnist.test.images))
    test_acc_list.append(test_acc)
    print ("test accuracy: %f" %(test_acc))

################# Training ##############################


test_acc_list = []
train_acc_list = []
# for i in range(70000):
for i in range(7000):
    batch = mnist.train.next_batch(128)
#     if i%1000 == 0: #every 1000 batches we save the model and output the train accuracy
    if i%100 == 0:
        print_test_accuracy(test_acc_list)
        saver.save(sess, checkpoint_path)
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_acc_list.append(train_accuracy)
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

################# Save Model ##############################


import pickle

with open('test_acc_list.p', 'wb') as fp:
    pickle.dump(test_acc_list, fp)

with open('test_acc_list.p', 'wb') as fp:
    pickle.dump(test_acc_list, fp)
