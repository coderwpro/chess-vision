import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import TFAutoModel, AutoModel

def  load_data(image_dir, labels_csv):
    images = []
    boxes = []
    labels = []
    data = pd.read_csv(labels_csv)

    for i, row in data.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        images.append(image)

def convert_ckpt_to_pt():
    checkpoint = torch.load("C:\\Users\\smirz\\OneDrive\\Documents\\Coding Minds\\Chess vision\\chess-vision\\checkpoint.ckpt",map_location=torch.device('cpu'))
    print(checkpoint.keys())
    state_dict = checkpoint['state_dict']
   # model_tf = TFAutoModel.from_pretrained("checkpoint.ckpt", from_tf=True)
    for key in state_dict.keys():
        print(f"Layer: {key}, Shape: {state_dict[key].shape}")
# Convert to PyTorch
    #model_pt = AutoModel.from_pretrained("checkpoint.ckpt")
    #model_pt.save(checkpoint.state_dict(),'model.pt')

convert_ckpt_to_pt()    