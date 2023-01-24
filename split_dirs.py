import argparse
import splitfolders
import os
import shutil


parser = argparse.ArgumentParser(description='Split directories to train, eval, test')
parser.add_argument('--path_to_imgs', metavar='DIR', nargs='?', default='datasets/augmented_dogs/',
                    help='path to dataset')
parser.add_argument('--destination_path', metavar='DIR', nargs='?', default='datasets/augmented_dogs_splitted/',
                    help='path to save augmented dataset')
parser.add_argument("--seed", default=42, type=int, help="Seed for random split")
parser.add_argument("--ratios", default=(.2, .05, 0.75), type=tuple, help="Ratios for split (train, eval, test)")
parser.add_argument("--fixed", default=(500, 500, 499), type=tuple, help="How many images to keep.")


if __name__ == '__main__':
    args = parser.parse_args()

    # if the destination path exists, remove it
    if os.path.exists(args.destination_path):
        print("Removing destination path...")
        shutil.rmtree(args.destination_path)

    if args.fixed:
        print("Splitting based on fixed numbers...")
        splitfolders.fixed(
            args.path_to_imgs, # The location of dataset
            output=args.destination_path, # The output location
            seed=args.seed, # The number of seed
            fixed=args.fixed,
            group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
            move=False # If you choose to move, turn this into True
        )
    else:
        print("Splitting based on ratios...")    
        splitfolders.ratio(
            args.path_to_imgs, # The location of dataset
            output=args.destination_path, # The output location
            seed=args.seed, # The number of seed
            ratio=args.ratios, # The ratio of splited dataset
            group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
            move=False # If you choose to move, turn this into True
        )