import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

def load_data(image_dir, labels_csv):
    images = []
    boxes = []
    labels = []
    data = pd.read_csv(labels_csv)

    for i, row in data.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224,224))
        images.append(images)

