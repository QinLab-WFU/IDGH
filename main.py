from train.hash_train import Trainer
from utils import get_args
import os

def get_class_num(name):
    r = {"coco": 80, "archive": 24, "nuswide": 21, "IAPR": 291}[name]
    return r

if __name__ == "__main__":

    args = get_args()

    for dataset in ["archive"]:

        print(f"processing dataset: {dataset}")
        for hash_bit in [16, 32, 64, 128]:
            args = get_args()
            # args.seed = seed
            args.save_dir = './result/'
            args.epochs = 101
            args.batch_size = 64
            if dataset == "nuswide":
                args.caption_file = "caption.txt"
            else:
                args.caption_file = "caption.mat"
            args.dataset = dataset
            args.n_classes = get_class_num(dataset)

            args.n_bits = hash_bit
            args.output_dim = hash_bit
            args.save_dir = os.path.join(args.save_dir, args.dataset, str(args.output_dim))
            Trainer(args)
