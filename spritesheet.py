##################################################
#                                                #
#   Removes background for all files without a   #
#   BG-less version and creates spritesheet      #
#                                                #
##################################################

from settings import *

import math
import subprocess
import numpy as np
import cv2
import os

SKIP_EXISTING = True
MAX_WIDTH = 16384
RESIZE = 1

def check_file(file):
    return file.endswith("_s.png") or file == SHEET_NAME or not file.endswith('.png')

imgs = []

raw_files = [f for f in os.listdir(CLEAN_PATH) if not check_file(f)]
for i in range(32):
    files = []
    hold = -1
    for i in range(len(raw_files)):
        if hold < 0:
            if i + 1 >= len(raw_files):
                files.append(raw_files[i])
                continue
            
            t1, t2 = raw_files[i].replace(".png", ""), raw_files[i + 1].replace(".png", "")
            if t1 in t2 or t2 in t1:
                cmp = len(t2) - len(t1)
                # print(f"1 comparing {t1} and {t2}, value {cmp}")
                if cmp < 0:
                    hold = i
                    files.append(raw_files[i + 1])
                else:
                    files.append(raw_files[i])
            else:
                files.append(raw_files[i])
        else:
            if i + 1 >= len(raw_files):
                files.append(raw_files[hold])
                continue
                
            t1, t2 = raw_files[hold].replace(".png", ""), raw_files[i + 1].replace(".png", "")
            if t1 in t2 or t2 in t1:
                cmp = len(t2) - len(t1)
                # print(f"2 comparing {t1} and {t2}, value {cmp}")
                if cmp < 0:
                    files.append(raw_files[i + 1])
                else:
                    files.append(raw_files[hold])
                    hold = -1
            else:
                files.append(raw_files[hold])
                hold = -1
    raw_files = files


for file in files:
    
    print(file)

    clean_img_path    = f"{CLEAN_PATH}/{file}"
    stripped_img_path = f"{CLEAN_PATH}/{file.replace('.png', '_s.png')}"

    if not SKIP_EXISTING or not os.path.exists(stripped_img_path):
        subprocess.call(["rembg", "i", clean_img_path, stripped_img_path])

    input_img = cv2.imread(stripped_img_path, cv2.IMREAD_UNCHANGED)
    imgs.append(cv2.resize(input_img, (int(input_img.shape[1]*RESIZE), int(input_img.shape[0]*RESIZE))))

row_count = 1
row_size = len(imgs)
h, w = imgs[0].shape[0], sum([img.shape[1] for img in imgs])
dy = h

new_w = w
while new_w > MAX_WIDTH:
    row_count *= 2
    row_size = math.ceil(len(imgs) / float(row_count))
    new_w = row_size * imgs[0].shape[0]
h, w = h * row_count, new_w

output_img = np.zeros((h, w, imgs[0].shape[2]))
img_i = 0
for row_i in range(row_count):
    x = 0
    for col_i in range(row_size):
        img = imgs[img_i]
        dx = img.shape[1]

        output_img[dy*row_i:dy*(row_i+1),x:x+dx] = img

        x += dx
        img_i += 1
        if img_i >= len(imgs):  break
    if img_i >= len(imgs):  break

cv2.imwrite(SHEET_PATH, output_img)