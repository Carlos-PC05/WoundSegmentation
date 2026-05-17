import torch

""" Rutas a los datos """
DATA_PATH = r"C:\Users\carlo\WoundSegmentation\data"
IMG_PATH1 = r"C:\Users\carlo\WoundSegmentation\data\HAM10000_images_part_1"
IMG_PATH2 = r"C:\Users\carlo\WoundSegmentation\data\HAM10000_images_part_2"
MASKS_PATH = r"C:\Users\carlo\WoundSegmentation\data\HAM10000_segmentations_lesion_tschandl"

""" Hiperparámetros de entrenamiento """
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

""" Parámetros de la imagen """
IMAGE_SIZE = 256