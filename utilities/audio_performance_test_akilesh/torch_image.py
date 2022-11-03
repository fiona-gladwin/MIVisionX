import torch
import torchvision
from PIL import Image
import cv2
import os
import timeit

tot_time =0 
for i in range(100):
    folder_path = "/media/audio_samples/MIVisionX-data/rocal_data/100sample/"
    file_list = os.listdir(folder_path)
    
    for fn in file_list:
        file_name = folder_path + fn 
        img = Image.open(file_name)
        if(not img):
            print("Error in image read ",file_name)
            exit(0)
        center_crop = torchvision.transforms.ColorJitter( brightness = 1.4 , contrast = 0.4 , saturation = 1.9, hue = 0.1 )
        start = timeit. default_timer()
        img = center_crop(img)
        tot_time = tot_time + (timeit. default_timer()-start)
print(tot_time*10000)  

