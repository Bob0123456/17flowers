# -*- coding: utf-8 -*- 
# author: liaoming
# python version: python2.7 and python3.6 is ok, please note the print function.
import tensorflow as tf 
import os 
import argparse 
from tensorflow.python.framework import graph_util 

def ckpt2pb(ckpt_dir, pb_dir, save_name, output_nodes, sep = ","):
    """
    params: \n
        ckpt_dir: The directory where stores the ckpt models. 
        pb_dir: The directory where to store the converted pb model. 
        save_name: the name used by the pb file. 
        output_nodes: the output nodes of the ckpt model. 
        sep: the seperator of the output_nodes string. 
    return: None, but it will write the pb model to disk. 
    """
    if not tf.gfile.Exists(pb_dir):
        tf.gfile.MakeDirs(pb_dir)
    # get checkpoint file
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = checkpoint.model_checkpoint_path # ckpt_path

    # define saver to restore the meta data.
    saver = tf.train.import_meta_graph( # 从文件中导入计算图
        meta_graph_or_file = ckpt_path + ".meta", clear_devices=True)
    default_graph = tf.get_default_graph() # to get the default graph in this thread.
    graph_def = default_graph.as_graph_def() # Returns a serialized `GraphDef` representation of this graph.

    with tf.Session() as sess: # 建立会话
        saver.restore(sess, ckpt_path)  # 从ckpt模型中恢复数据
        # print("graph_def:")
        # print("***********************************")
        # for op in default_graph.get_operations():
        #     print(op.name, "<->", op.values())
        output_graph_def = graph_util.convert_variables_to_constants(
            sess = sess,
            input_graph_def = graph_def,
            output_node_names = output_nodes.split(sep))
        # write to pb model.
        with tf.gfile.GFile(os.path.join(pb_dir, save_name), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        
def main():
    argparser = argparse.ArgumentParser()
    # define input arguments
    argparser.add_argument("ckpt_dir", type = str, help = "The directory where stores the ckpt model")
    argparser.add_argument("pb_dir", type = str, help = "The directory will to store the pb model")
    argparser.add_argument("save_name", type = str, help = "The save name of generated pb model")
    argparser.add_argument("output_nodes", type = str, help = "The output nodes defined int the ckpt model, split by ${sep}")
    argparser.add_argument("sep", type = str, nargs = "?", default = ',', help = "The seprator of output_nodes")
    # get input arguments
    args = argparser.parse_args()
    ckpt2pb(args.ckpt_dir, args.pb_dir, args.save_name, args.output_nodes, args.sep)

if __name__ == "__main__":
    main()
