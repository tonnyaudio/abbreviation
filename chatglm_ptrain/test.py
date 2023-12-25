import argparse
from pprint import pprint

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=4e-5)

    return parser.parse_args()

if __name__ == '__main__':

    args = set_args()
    pprint(args)
    print(args)
    # config = RankerConfig(**vars(args))

    batch_size=args.batch_size
    num_epochs=args.num_epochs
    lr=args.lr
    print(batch_size,num_epochs,lr)
    