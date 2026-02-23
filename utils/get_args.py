import argparse
import math
import os

import xlrd


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-layer", type=str, default="linear", help="choice a hash layer [select, linear] to run. select: select mechaism, linear: sign function.")
    parser.add_argument("--save-dir", type=str, default="./result/")
    parser.add_argument("--clip-path", type=str, default="./ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--dataset", type=str, default="archive_v", help="choise from [coco, archive_v, nuswide, IAPR, imagenet]")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")

    parser.add_argument("--output-dim", type=int, default=16)

    parser.add_argument("--HM", type=int, default=500)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--alpha", type=int, default=1)

    parser.add_argument("--tau", type=int, default=0.1)

    parser.add_argument("--epochs", type=int, default=101)
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--query-num", type=int, default=5000)

    parser.add_argument("--train-num", type=int, default=10000)
    parser.add_argument("--lr-decay-freq", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1814)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--clip-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    parser.add_argument("--n_bits", type=int, default=16, help="length of hashing binary")
    parser.add_argument("--n_classes", type=int, default=21, help="number of dataset classes")

    parser.add_argument("--margin", type=float, default=1.0, help="None")
    parser.add_argument("--scaling_p", type=float, default=1.0, help="None")
    parser.add_argument("--scaling_x", type=float, default=3.0, help="None")
    parser.add_argument("--h_dim", type=int, default=0, help="None")
    parser.add_argument("--prior", type=int, default=0, help="None")
    parser.add_argument("--eta", type=float, default=100, help="None")

    parser.add_argument("--eta_e", type=float, default=0.01, help="None")

    parser.add_argument("--smooth_factor", type=float, default=0.15, help="None")
    parser.add_argument("--lam", type=float, default=0.01, help="None")
    parser.add_argument("--m", type=float, default=1, help="None")
    parser.add_argument("--model_lr", type=float, default=0.0001, help="None")
    parser.add_argument("--proxy_nca_lr", type=float, default=0.015, help="None")
    parser.add_argument("--embedding_lr", type=float, default=0.001, help="None")

    parser.add_argument("--trainer", type=str, default="ProgCoPL", help="name of trainer")

    parser.add_argument("--intra", default=1, type=int)
    parser.add_argument("--start_epoch", default=31, type=int)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--intra_lamda", default=0.8, type=float)
    parser.add_argument("--aug_num", default=3, type=int)

    parser.add_argument("--long_tail_IF", default=100, type=int)

    parser.add_argument('--loss', default='MS', type=str)

    parser.add_argument("--momentum", default=0.2, type=float)

    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument("--num_neighbor", default=25, type=int)

    parser.add_argument("--diag", default=1, type=int)
    parser.add_argument("--global_update", default=1, type=int)
    parser.add_argument("--update_every_M_epoch", default=2, type=int)

    parser.add_argument("--noise-rate", type=float, default=0.0)  # noise 0.2 0.5 0.8

    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help='mixup interpolation coefficient (default: 1.0)')

    parser.add_argument('--save_feature_interval', type=int, default=5,
                        help='Save features every N epochs')
    parser.add_argument('--max_feature_samples', type=int, default=1000,
                        help='Maximum number of samples to save per epoch')

    args = parser.parse_args()

    args.method = 'IDGH'

    return args

args = get_args()
