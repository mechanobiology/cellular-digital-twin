from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import time
import os
import sys
from aae import GAN

def configure():
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("updates_per_epoch", 900, "number of updates per epoch")
    flags.DEFINE_integer("max_epoch", 50, "max epoch for total training")
    flags.DEFINE_integer("max_generated_imgs", 1, "max generated imgs for each input")
    flags.DEFINE_integer("max_test_epoch", 10, "max  test epoch")
    flags.DEFINE_integer("summary_step", 100, "save summary per #summary_step iters")
    flags.DEFINE_integer("save_step", 900, "save model per #save_step iters")
    flags.DEFINE_integer("n_class", 10, "number of classes")
    flags.DEFINE_float("learning_rate_dis", 2e-4, "learning rate discriminator")
    flags.DEFINE_float("learning_rate_gen", 2e-3, "learning rate generator")
    flags.DEFINE_float("gamma_gen", 1e-4, "gamma ratio for generator loss")
    flags.DEFINE_float("gan_noise", 0.01, "injection noise for the GAN")
    flags.DEFINE_bool("noise_bool", False, "add noise on all GAN layers")
    flags.DEFINE_string("working_directory", 'ML_framework/evaluation_Nucl',
                        "the file directory where predictions are written")
    flags.DEFINE_integer("hidden_size", 16, "size of the hidden VAE unit")
    flags.DEFINE_integer("checkpoint", 50, "number of epochs to be reloaded")
    flags.DEFINE_integer("height", 256, "height of image") #mod
    flags.DEFINE_integer("width", 256, "width of image") #mod
    flags.DEFINE_integer("slices", 64, "number of slices") #mod
    flags.DEFINE_integer("train_range", 0, "number of samples for training")
    flags.DEFINE_integer("test_range", 10, "number of samples for testing")
    flags.DEFINE_integer("mode", 2, "mode determining which structure to train/predict: use 1 for FA and 2 for nucleus")
    flags.DEFINE_string("modeldir", 'ML_framework/checkpoint_Nucl', "the model directory")
    flags.DEFINE_string("logdir", 'ML_framework/logdir_Nucl', "the log directory")
    flags.DEFINE_string("sampledir", 'ML_framework/sampledir_Nucl', "the sample directory")
    flags.DEFINE_string("datadir", 'ML_framework/data/3d_fa_ind_test_nucl.h5', "the directory containing the data")
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS
 
def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        dest='action',
        type=str,
        default='train',
        help='actions: train, or independent_test')
    args = parser.parse_args()
    if args.action not in ['train', 'independent_test']:
        print('invalid action: ', args.action)
        print("Please input a action: train, independent_test")
    else:
        model= GAN(tf.Session(),configure())
        getattr(model,args.action)()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()