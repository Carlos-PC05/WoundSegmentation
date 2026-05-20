from utils import DiceLoss
import config as c

for epoch in range(c.NUM_EPOCHS):
    for batch in dataloader_train