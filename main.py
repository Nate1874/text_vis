from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import time
import os
import sys
from model import CNN

def configure():
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 64, "batch size")
    flags.DEFINE_integer("max_epoch", 500, "max epoch for total training")
    flags.DEFINE_integer("opt_iter", 20000, "the max number of optimization iteration")
    flags.DEFINE_integer("sequence_length", 195, "sequence_length")
    flags.DEFINE_string("type", "sp_location", "The target to be visualized, can be layer, sp_location, neuron, channel")
    flags.DEFINE_integer("layer_num", 1, "which layer, 1, or 2 or 3")
    flags.DEFINE_integer("channel_num", 0, "which channel in the layer")
    flags.DEFINE_integer("x", 0, "spatial location-- x")
    flags.DEFINE_integer("num_classes", 4, "num_classes")
    flags.DEFINE_integer("embed_size", 300, "embed_size")    
    flags.DEFINE_integer("vocab_size", 84252, "vocab size")#84252
    flags.DEFINE_integer("summary_step", 100, "save summary per #summary_step iters")
    flags.DEFINE_integer("save_step", 1000, "save model per #save_step iters")
    flags.DEFINE_integer("eva_step", 1, "save model per #save_step iters")
    flags.DEFINE_float("learning_rate", 2e-4, "learning rate")
    flags.DEFINE_boolean("enable_word_embeddings", True, "if use pre-trained embedding")
    flags.DEFINE_boolean("isTraining", False, "if it is training or interpreting")
    flags.DEFINE_integer("checkpoint", 66666, "number of epochs to be reloaded")
    flags.DEFINE_integer("d_model", 512, "dimension in the model")
    flags.DEFINE_string("modeldir", './modeldir_fake', "the model directory")
    flags.DEFINE_string("logdir", './logdir_fake', "the log directory")
    flags.DEFINE_string("sampledir", './sampledir', "the sample directory")
    flags.DEFINE_string("path_embedding", '/tempspace/hyuan/data_text/GoogleNews-vectors-negative300.bin', "the path for pre-trained embedding")
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        dest='action',
        type=str,
        default='train',
        help='actions: train, or test')
    args = parser.parse_args()
    if args.action not in ['train', 'test']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test")
    else:
        model= CNN(tf.Session(),configure())
        getattr(model,args.action)()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    tf.app.run()





