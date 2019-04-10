# -*- coding: utf-8 -*-
import tensorflow as tf 
import infer 
import tf_records as tr # TFRecords
import os 
import numpy as np 
import time # for check time consumption
from train import args
import train 

# set log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

eval_max_steps = args.max_eval_iters
eval_batch_size = args.eval_batch_size

def evaluate(net, tfr_path):
    # define graph
    with tf.Graph().as_default() as graph:
        # def images and labels placeholder
        xin = tf.placeholder(dtype = tf.float32,
            shape = [eval_batch_size, infer.IMAGE_SIZE, infer.IMAGE_SIZE, infer.NUM_CHANNELS],
            name = "xin")
        # yin = tf.placeholder(dtype = tf.float32,
        #     shape = [eval_batch_size, infer.NUM_LABELS],
        #     name = "yin")
        # define images and labels iterator 
        images, labels = tr.read_tfrecords_by_data(tfr_path, (224, 224), 3, batch_size = eval_batch_size)
        # define net 
        yout = net.inference(xin, train = False, regularizer = None)
        scores = tf.nn.softmax(yout) # the score of each class.
        classes = tf.argmax(scores, 1)
        # compute accurary
        
        # define saver to restore netwrok from file
        saver = tf.train.Saver() # restore all variables

        # do evaluation
        pred_right = 0
        pred_total = 0
        with tf.Session() as sess: # get session.
            # get ckpt
            ckpt = tf.train.get_checkpoint_state(train.MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path: # check model and model_path is exist.
                # load model
                saver.restore(sess, ckpt.model_checkpoint_path) # restore variables.
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1]
                # get accuracy
                for _ in range(int(args.max_eval_iters / eval_batch_size)):
                    image_batch, label_batch = sess.run([images, labels])
                    pred_scores, pred_classes = sess.run([scores, classes], feed_dict = {xin: image_batch}) # if you use the dropout layer, check whether should feed a keep probbility param.
                    # print(pred_scores) # if do evaluation, you need not to print info like that.
                    # print(pred_classes)
                    # print(label_batch)
                    equal = tf.equal(pred_classes, label_batch).eval()
                    pred_right += sum(equal)
                    pred_total += len(equal)
                    # print("Accuracy: {:.4f}".format(float(pred_right)/pred_total))
                val_accu = float(pred_right) / pred_total
                print("After {} training step(s), VAL ACCURACY: {:.4f}".format(global_step, val_accu))
                return val_accu
            else:
                print("No checkpoint file found! Pleace check again.")
                return

def main(argv = None):
    print("Evaluation start...")
    val_accu = evaluate(infer, args.eval)
    print("Evaluation done...")

main() 