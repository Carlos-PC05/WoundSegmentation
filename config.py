""" Rutas a los datos """
# Cambiar dependiendo de donde se encuentren los datos en tu máquina
DATA_PATH = r"C:\Users\CARLOS\WoundSegmentation\data"
IMG_PATH1 = r"C:\Users\CARLOS\WoundSegmentation\data\HAM10000_images_part_1"
IMG_PATH2 = r"C:\Users\CARLOS\WoundSegmentation\data\HAM10000_images_part_2"
MASKS_PATH = r"C:\Users\CARLOS\WoundSegmentation\data\HAM10000_segmentations_lesion_tschandl"

""" Hiperparámetros de entrenamiento """
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_EPOCHS = 10

""" Parámetros de la imagen """
IMAGE_SIZE = 256