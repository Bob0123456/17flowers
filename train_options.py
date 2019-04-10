# -*- coding: utf-8 -*-
# author: liaoming
# define training options for this model.

import argparse

parser = argparse.ArgumentParser()

# define options
# for data
parser.add_argument("--train", required = True, type = str, 
    help = "Path to the train data.")
parser.add_argument("--eval", required = True, type = str,
    help = "Path to the eval data.")

# for model

# for learning(training)
parser.add_argument("--max_train_iters", type = int, default = 100000,
    help = "Max training iterations.")
parser.add_argument("--lr", type = float, default = 0.00001,
    help = "Base learning rate.")
parser.add_argument("--lr_decay", type = float, default = 0.99,
    help = "The decay param of learning rate.")
parser.add_argument("--batch_size", type = int, default = 64,
    help = "Batch size used for training model.")
parser.add_argument("--weight_decay", type = float, default = 0.0002,
    help = "The weight decay used for regularization.")
parser.add_argument("--moving_ave_decay", type = float, default = 0.99,
    help = "The moving average decay param for training the moving average model.")

# for evaluation
parser.add_argument("--max_eval_iters", type = int, default = 100,
    help = "Max evaluation iterations.")
parser.add_argument("--eval_batch_size", type = int, default = 1,
    help = "Batch size used")

# for experiment
parser.add_argument("--model_dir", type = str, default = "models",
    help = "Path to model folder, where stores the trained folder.")
parser.add_argument("--save_model_name", type = str, default = "model",
    help = "Checkpoint name to save.")
parser.add_argument("--checkpoint_iters", type = int, default = 2000,
    help = "Model checkpoint iter for saveing model.")
parser.add_argument("--eval_iters", type = int, default = 2000,
    help = "Evaluation period.")

    