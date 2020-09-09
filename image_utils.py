import os
import re
import cv2
import numpy as np


IMG_HEIGHT, IMG_WIDTH = 240, 320


def load_images(LABEL_MAP, data_dir, img_height, img_width):
    images = []
    labels = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        match = re.match(r"(\d+)_(\d+).bmp", img_name)
        if match:
            _, img_label = match.groups()
            img_label = int(img_label)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_height, img_width))
            images.append(img)
            labels.append(LABEL_MAP[img_label])
    return np.array(images), np.array(labels)


def get_img_and_labels(data_dir):
    images, labels = load_images(data_dir, IMG_HEIGHT, IMG_WIDTH)
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    index_to_label = {index: label for index, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_index[label] for label in labels])
    return images, numeric_labels, index_to_label


def save_resized_images_as_bmp(images, labels, data_dir, img_height, img_width):
    for i, (image, label) in enumerate(zip(images, labels)):
        img_resized = cv2.resize(image, (img_height, img_width))
        img_name = f"{i}_{label}.bmp"
        img_path = os.path.join(data_dir, img_name)
        cv2.imwrite(img_path, img_resized)
