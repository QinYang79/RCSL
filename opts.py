import argparse
import random
import os

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--tau', default=0.03, type=float)
    parser.add_argument('--stage', default='learning', type=str)

    parser.add_argument('--mining_start', default=10, type=int)
    parser.add_argument('--paired_length', default=5000, type=int)
 
    parser.add_argument('--init_txt', default='uniform',
                        help='uniform|glove only in GRU')
    parser.add_argument('--img_enhance', action='store_false',
                        help='Default is True')
    parser.add_argument('--caption_enhance', action='store_false',
                        help='Default is True')
    parser.add_argument('--use_bi_gru', action='store_false',
                        help='Default is True')
    parser.add_argument('--logger_path', default='./runs/test/checkpoint',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_path', default='./runs/test/log',
                        help='Path to save the model.')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--data_path', default='/home/qinyang/projects/data/cross_modal_data/data/data',
                        help='path to datasets')
    parser.add_argument('--vocab_path', default='/home/qinyang/projects/data/cross_modal_data/data/vocab',
                        help='Path to saved vocabulary json files.')

    parser.add_argument('--MaxEpoch', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--MineEpoch', default=25, type=int)
    parser.add_argument('--UpdateEpoch', default=35, type=int)
                                      
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0005, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    return parser 
