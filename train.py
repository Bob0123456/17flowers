# -*- coding: utf-8 -*- 
import tensorflow as tf 
import infer
import os 
import numpy as np 
import tf_records

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# SETTINGS
BATCH_SIZE = 8
LEARNING_RATE_BASE = 0.00001
LEARNING_RATE_DECAY = 0.99
TRAIN_STEPS = 30000
# 
REGULARAZTION_RATE = 0.0002
# 
MOVING_AVE_DECAY = 0.99 

MODEL_PATH = "./models/"
MODEL_NAME = "LeNet_5_c17_flowers.ckpt"

def train():
    # place holder
    xin = tf.placeholder(dtype = tf.float32,
        shape=[BATCH_SIZE, infer.IMAGE_SIZE, infer.IMAGE_SIZE, infer.NUM_CHANNELS],
        name = "xin")
    yin = tf.placeholder(dtype = tf.float32,
        shape=[BATCH_SIZE, infer.NUM_LABELS],
        name = "yin")
    global_step = tf.Variable(0, trainable = False)
    # regulazation
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # compute graph
    yout = infer.inference(input_tensor = xin, train = True, regularizer = regularizer)
    print ("yout.shape: {}".format(yout.shape))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = yout,
        labels = tf.argmax(yin, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer_loss = tf.add_n(tf.get_collection("losses"))
    # total loss
    loss = cross_entropy_mean + regularizer_loss 

    learning_rate = tf.train.exponential_decay(
        learning_rate = LEARNING_RATE_BASE,
        global_step = global_step,
        decay_steps = 100,
        decay_rate = LEARNING_RATE_DECAY)
    #  TRAIN STEP
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    train_accu = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(yin, 1), tf.argmax(yout, 1)), tf.float32)
    )

    saver = tf.train.Saver()

    tfr_path = "./17flowers.tfr"
    # image, label = tf_records.read_tfrecords(tfr_path, (224, 224), 3)
    # images, labels = tf_records.get_batch(image, label, BATCH_SIZE, capacity = 16 * BATCH_SIZE, min_after_dequeue = 4 * BATCH_SIZE)
    images, labels = tf_records.read_tfrecords_by_data(tfr_path, (224, 224), 3, batch_size=BATCH_SIZE)
    labels = tf.one_hot(labels, infer.NUM_LABELS, 1, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        print ("images.shape: {}, labels: {}".format(images.shape, labels))
        print ("training...")
        # train loop
        for i in range(TRAIN_STEPS):
            # print "step {} ".format(i), 
            img_batch, label_batch = sess.run([images, labels])
            # print "image_batch.shape:", img_batch.shape 
            # label_batch = tf.one_hot(label_batch, 17, 1, 0)
            train_feed = {xin: img_batch.astype(np.float32), yin: label_batch}
            _, loss_value, step, lr = sess.run([train_step, loss, global_step, learning_rate], feed_dict = train_feed)
            if (i + 1) % 10 == 0:
                accuracy = sess.run(train_accu, feed_dict = {xin: img_batch, yin: label_batch})
                print ("after {} step(s), loss (on batch): {:.4f}, lr: {:.8f}, train_accuracy: {:.3f}".format(step, loss_value, lr, accuracy))
            if (i + 1) % 1000 == 0:
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step = global_step)
                print ("model saved to {}".format(os.path.join(MODEL_PATH, MODEL_NAME)))
        # coord.request_stop()
        # coord.join(threads)

def main(argv = None):
    print ("main function begin!")
    train()
    print ("optimize done!")

if __name__ == "__main__":
    main() 
