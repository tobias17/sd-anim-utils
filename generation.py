##################################################
#                                                #
#   Very first file to run to init a workspace   #
#                                                #
##################################################

from settings import *

import subprocess
import numpy as np
import json
import cv2
import os

if not os.path.exists(WORKSPACE):  os.makedirs(WORKSPACE)

assert os.path.exists(POSES_FOLDER),   f"poses folder must exist, settings.py says it is '{POSES_FOLDER}'"
assert os.path.exists(TURNTABLE_PATH), f"turntable image must exist, settings.py says it is '{TURNTABLE_PATH}'"
assert os.path.exists(SOURCE_PATH),     f"source image must exist, settings.py says it is '{SOURCE_PATH}'"
subprocess.call(["rembg", "i", SOURCE_PATH, SOURCE_S_PATH])
assert os.path.exists(SOURCE_S_PATH),   f"error creating stripped source image"

imgs = [f for f in os.listdir(POSES_FOLDER) if f != TURNTABLE_NAME]
print(f"found pose images: {imgs}")

tt = cv2.imread(TURNTABLE_PATH)
assert tt is not None, f"error loading turntable image '{TURNTABLE_PATH}', cv2 returned null"

source = cv2.imread(SOURCE_S_PATH, cv2.IMREAD_UNCHANGED)
assert source is not None, f"error loading source image '{SOURCE_S_PATH}', cv2 returned null"

assert tt.shape[0] == source.shape[0], f"source and turntable images must have the same height, {source.shape[0]} != {tt.shape[0]}"
assert tt.shape[1] == source.shape[1], f"source and turntable images must have the same width, {source.shape[1]} != {tt.shape[1]}"

iter_name = f"{ITER_PREFIX}000"

first = True
for img in imgs:
    assert img.endswith(".png"), f"pose images must be of type png, invalid for {img}"

    path = f"{POSES_FOLDER}/{img}"
    i = cv2.imread(path)

    assert i is not None, f"error loading pose image '{path}', cv2 returned null"
    assert tt.shape[0] == i.shape[0], f"turntable and pose images must have the same height, was not the case for {img}: {tt.shape[0]} != {i.shape[0]}"
    assert tt.shape[2] == i.shape[2], f"turntable and pose images must have the same number of color channels, was not the case for {img}: {tt.shape[2]} != {i.shape[2]}"

    pose = np.zeros((tt.shape[0], tt.shape[1] + i.shape[1], tt.shape[2]))
    pose[:, :tt.shape[1]] = tt
    pose[:, tt.shape[1]:] = i

    output = np.ones(pose.shape)
    output *= BG_COLOR
    for c in range(3):
        output[:, :tt.shape[1], c] = (1-source[:,:,3]/255)*output[:,:tt.shape[1],c] + (source[:,:,3]/255)*source[:,:,c]
    
    if first:
        source_g = output[:, :tt.shape[1]]
        cv2.imwrite(SOURCE_G_PATH, source_g)
        first = False

    folder = f'{WORKSPACE}/{iter_name}/{img.replace(".png", "")}'
    if not os.path.exists(folder):  os.makedirs(folder)

    cv2.imwrite(f"{folder}/{POSE_IMG_NAME}", pose)
    cv2.imwrite(f"{folder}/{INPUT_IMG_NAME}", output)
    with open(f"{folder}/{DATA_NAME}", "w") as f:
        f.write(json.dumps({ "start": tt.shape[1], "end": tt.shape[1] + i.shape[1] }))
    
with open(f"{WORKSPACE}/{iter_name}/{TOUCH_FILE_NAME}", "w") as f:
    f.write("_")
