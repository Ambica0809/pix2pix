import torch
import generator_model as gen
from PIL import Image
import numpy as np
import config
from torchvision.utils import save_image

def main():
    model=gen.Generator(in_channels=3, features=64)
    ld=torch.load("gen.pth.tar")
    model.load_state_dict(ld['state_dict'])
    img_path="maps/val/9.jpg"
    image = np.array(Image.open(img_path))
    input_image = image[:, :600, :]
    augmentations = config.both_transform(image=input_image)
    tr_input_image = augmentations["image"]
    
    tr_input_image=tr_input_image.view(1,tr_input_image.shape[0],tr_input_image.shape[1],tr_input_image.shape[2])
    
    model.eval()
    with torch.no_grad():
        y_fake = model(tr_input_image)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        save_image(y_fake,f"pred.png")
        save_image(tr_input_image * 0.5 + 0.5,f"inp.png")

if __name__ == "__main__":
    main()