import torch
import torchvision
import config as c 
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import os

class SkinLesionDataset(Dataset):

    def __init__(self):
        #Load CSV, store routes, transforms...
        super().__init__()
        self.pairs = []
        self.img_path = ""

        #Paths
        self.csv_path = os.path.join(c.DATA_PATH, 'HAM10000_metadata.csv')
        self.img1_path = os.path.join(c.DATA_PATH, 'HAM10000_images_part_1')
        self.img2_path = os.path.join(c.DATA_PATH, 'HAM10000_images_part_2')

        #Transforms
        self.transform = None #for now
        
        #csv dataframe
        self.df = pd.read_csv(self.csv_path)        

        for image_id in self.df['image_id']:
            img_filename = image_id + '.jpg'
            mask_filename = image_id + '_segmentation.png'

            #Search in part1, if not, part2
            path_part1 = os.path.join(self.img1_path, img_filename)
            path_part2 = os.path.join(self.img2_path, img_filename)

            if os.path.exists(path_part1):
                img_path = path_part1
            else: 
                img_path = path_part2

            mask_path = os.path.join(c.MASKS_PATH, mask_filename)

            pair = (img_path, mask_path)
            self.pairs.append((pair))

    def __len__(self):
        #Return how many rows has the dataset
        return len(self.pairs)

    def __getitem__(self, index):
        #Given an index, returns a sample (image, mask)
        img_path, mask_path = self.pairs[index]

        #Open the image and mask, apply transforms, convert to tensor and return
        img = Image.open(img_path).convert('RGB') #color img
        img = TF.resize(img, (c.IMAGE_SIZE, c.IMAGE_SIZE))

        mask = Image.open(mask_path).convert('L') #Gray scale
        mask = TF.resize(mask, (c.IMAGE_SIZE, c.IMAGE_SIZE))

        torchIMG = TF.to_tensor(img)
        torchMask = TF.to_tensor(mask)

        torchPair = (torchIMG, torchMask)

        return torchPair
        


