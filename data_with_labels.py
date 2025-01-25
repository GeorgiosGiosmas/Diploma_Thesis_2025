import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--image_dir',
    default="Data/images/train",
    help='Image directory')
parser.add_argument(
    '--label_dir',
    default="Data/labels/train",
    help='Labels\' directory')
parser.add_argument(
    '--dest_img_dir',
    default='DataNew/images',
    help='Destination image directory')
parser.add_argument(
    '--dest_label_dir',
    default='DataNew/labels',
    help='Destination labels\' directory')
parser.add_argument(
    '--split_ratio',
    default=0.7,
    type=float,
    help='Split ratio between training and validation dataset')

args, _ = parser.parse_known_args()

def train_test_split_labeled_images(img_dir, txt_dir, dest_img_dir, dest_txt_dir, split_ration=0.7):

    img_files = os.listdir(img_dir)
    lbl_files = os.listdir(txt_dir)

    impaths = []
    for item in img_files:
        impaths+=[os.path.join(img_dir, item)] 

    txtpaths = []
    for item in lbl_files:
        txtpaths+=[os.path.join(txt_dir, item)]

    for item in txtpaths:
        image_name = os.path.basename(item).split('.')[0]
        image_label = np.loadtxt(f"{txt_dir}/{image_name}.txt")
        image = cv2.imread(f"{img_dir}/{image_name}.png")
        cv2.imwrite(f"{dest_img_dir}/{image_name}.png" , image)
    

if __name__ == '__main__':
    
    train_test_split_labeled_images(args.image_dir, args.label_dir, args.dest_img_dir, args.sest_label_dir, args.split_ratio)


