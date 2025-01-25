import os
import random
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
    default='data_windturbines/images',
    help='Destination image directory')
parser.add_argument(
    '--dest_label_dir',
    default='data_windturbines/labels',
    help='Destination labels\' directory')
parser.add_argument(
    '--split_ratio',
    default=0.7,
    type=float,
    help='Split ratio between training and validation dataset')

args, _ = parser.parse_known_args()

def train_test_split(image_dir, label_dir, dest_img_dir, dest_labels_dir, split_ratio=0.7):
    
    # List the contents of data/label folders
    data_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    # Seperate the labelled images from the unlabelled ones
    labelled_images = []
    unlabelled_images = []
    for image in data_files:
        if os.path.basename(image).split('.')[0] +'.txt' in label_files:
            labelled_images.append(image)
        else:
            unlabelled_images.append(image)
    
    # Random shuffling to the images
    random.shuffle(labelled_images)
    random.shuffle(labelled_images)
    random.shuffle(unlabelled_images)
    random.shuffle(unlabelled_images)
    
    # The split ratio of the data to train and test, default value = 0.7
    split_index_labelled_images = int(len(labelled_images) * split_ratio)
    split_index_unlabelled_images = int(len(unlabelled_images) * split_ratio)
    
    train_labelled_data = labelled_images[:split_index_labelled_images]
    val_labelled_data = labelled_images[split_index_labelled_images:]

    train_unlabelled_data = unlabelled_images[:split_index_unlabelled_images]
    val_unlabelled_data = unlabelled_images[split_index_unlabelled_images:]
    
    # Save the labelled data to the train folder
    for image in train_labelled_data:
        image_name = os.path.basename(image).split('.')[0]
        image_label = np.loadtxt(f"{label_dir}/{image_name}.txt")
        image_ = cv2.imread(f"{image_dir}/{image}")
        cv2.imwrite(f"{dest_img_dir}/train/{image_name}.png" , image_)
        
        if type(image_label[0])==np.float64:
            image_label = np.array([image_label])
            np.savetxt(f"{dest_labels_dir}/train/{image_name}.txt", image_label)
        else:
            np.savetxt(f"{dest_labels_dir}/train/{image_name}.txt", image_label)

    # Save the labelled data to the val folder
    for image in val_labelled_data:
        image_name = os.path.basename(image).split('.')[0]
        image_label = np.loadtxt(f"{label_dir}/{image_name}.txt")
        image_ = cv2.imread(f"{image_dir}/{image}")
        cv2.imwrite(f"{dest_img_dir}/val/{image_name}.png" , image_)
        
        if type(image_label[0])==np.float64:
            image_label = np.array([image_label])
            np.savetxt(f"{dest_labels_dir}/val/{image_name}.txt", image_label)
        else:
            np.savetxt(f"{dest_labels_dir}/val/{image_name}.txt", image_label)

    # Save the unlabelled data to the train folder
    for image in train_unlabelled_data:
        image_name = os.path.basename(image).split('.')[0]
        image_ = cv2.imread(f"{image_dir}/{image}")
        cv2.imwrite(f"{dest_img_dir}/train/{image_name}.png" , image_)

    # Save the unlabelled data to the val folder
    for image in val_unlabelled_data:
        image_name = os.path.basename(image).split('.')[0]
        image_ = cv2.imread(f"{image_dir}/{image}")
        cv2.imwrite(f"{dest_img_dir}/val/{image_name}.png" , image_)

    
if __name__ == '__main__':

    train_test_split(args.image_dir, args.label_dir, args.dest_img_dir, args.dest_label_dir, args.split_ratio)
    
