##################################################
#                                                #
#   Cleans the extraction run and adds back in   #
#   a stable background                          #
#                                                #
##################################################

from settings import *

import subprocess
import numpy as np
import cv2
import os

if not os.path.exists(CLEAN_PATH):  os.mkdir(CLEAN_PATH)

for file in os.listdir(EXTRACT_PATH):
    if file.endswith("_s.png") or file == SHEET_NAME or not file.endswith('.png'):  continue
    print(file)

    clean_img_path    = f"{EXTRACT_PATH}/{file}"
    stripped_img_path = f"{CLEAN_PATH}/{file.replace('.png', '_s.png')}"
    subprocess.call(["rembg", "i", clean_img_path, stripped_img_path])
    
    img = cv2.imread(stripped_img_path, cv2.IMREAD_UNCHANGED)
    output = np.ones((img.shape[0], img.shape[1], 3))
    output *= BG_COLOR
    for c in range(3):
        output[:, :, c] = (1-img[:,:,3]/255)*output[:,:,c] + (img[:,:,3]/255)*img[:,:,c]
    cv2.imwrite(f"{CLEAN_PATH}/{file}", output)
