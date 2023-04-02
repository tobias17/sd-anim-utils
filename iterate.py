##################################################
#                                                #
#   Run an iteration on your workspace           #
#                                                #
##################################################

from settings import *

import random
import numpy as np
import json
import cv2
import os

def folder_name(i):
    return f"{WORKSPACE}/{ITER_PREFIX}{i:03}"

curr_folder = ""
next_folder = ""
for i in range(1000):
    next_folder = folder_name(i)
    if not os.path.exists(f"{next_folder}/{TOUCH_FILE_NAME}"):
        curr_folder = folder_name(i-1)
        break

print(f"running on: {curr_folder}")
print(f"generating: {next_folder}")
assert len(curr_folder) > 0, "could not find an iter folder, run generation.py to first create a workspace"

if not os.path.exists(next_folder):
    os.makedirs(next_folder)

if KEEP_TURNTABLE:
    assert os.path.exists(SOURCE_G_PATH),  f"turntable source image not found, expected file at '{SOURCE_G_PATH}'"
    assert os.path.exists(TURNTABLE_PATH), f"turntable pose image not found, expected file at '{TURNTABLE_PATH}'"
    turntable_pair = [ cv2.imread(SOURCE_G_PATH), cv2.imread(TURNTABLE_PATH) ]

img_pose_pairs = {}

root_folders = [f for f in os.listdir(curr_folder) if f != TOUCH_FILE_NAME]
exclusion_list = [DATA_JSON_NAME, INPUT_IMG_NAME, POSE_IMG_NAME]

for root_folder in root_folders:
    root = f"{curr_folder}/{root_folder}"

    data_path = f"{root}/{DATA_JSON_NAME}"
    pose_path = f"{POSES_FOLDER}/{root_folder}.png"
    input_paths = [f for f in os.listdir(root) if f not in exclusion_list]
    assert len(input_paths) == 1, f"expected 1 extra file in {root_folder}, got {len(input_paths)}: {input_paths}"
    input_path = f"{root}/{input_paths[0]}"

    assert os.path.exists(data_path),  f"could not find data_path for {root_folder}, searched {data_path}"
    assert os.path.exists(pose_path),  f"could not find pose_path for {root_folder}, searched {pose_path}"
    assert os.path.exists(input_path), f"could not find input_path for {root_folder}, searched {input_path}"

    with open(data_path) as f:
        data = json.loads(f.read())
    assert "start" in data, f"expected 'start' key in data json file for {root_folder} at {data_path}"
    assert "end" in data,   f"expected 'end' key in data json file for {root_folder} at {data_path}"

    input = cv2.imread(input_path)
    assert input is not None, f"input img was None for '{input_path}', expected value"
    assert input.shape[1] >= data["end"], f"input img must have width of at least data.json['end'], found {input.shape[1]} < {data['end']}"
    input = input[:,data["start"]:data["end"]]

    pose = cv2.imread(pose_path)
    assert pose is not None, f"pose img was None for '{pose_path}', expected value"

    assert input.shape[0] == pose.shape[0], f"input and pose need to have height, got {input.shape[0]} != {pose.shape[0]}"
    assert input.shape[1] == pose.shape[1], f"input and pose need to have width, got {input.shape[1]} != {pose.shape[1]}"
    assert input.shape[2] == pose.shape[2], f"input and pose need to have color depth, got {input.shape[2]} != {pose.shape[2]}"

    img_pose_pairs[root_folder] = [input, pose]

keys = list(img_pose_pairs.keys())
if RANDOMIZE_ORDER:
    random.seed(curr_folder + next_folder)
    random.shuffle(keys)
print(f"keys order: {keys}")

assert type(IMGS_LEFT) == type(0),  f"settings.IMGS_LEFT must be of type int, got {type(IMGS_LEFT)}"
assert type(IMGS_RIGHT) == type(0), f"settings.IMGS_RIGHT must be of type int, got {type(IMGS_RIGHT)}"
assert IMGS_LEFT >= 0,  f"settings.IMGS_LEFT must be non-negative, got {IMGS_LEFT}"
assert IMGS_RIGHT >= 0, f"settings.IMGS_RIGHT must be non-negative, got {IMGS_RIGHT}"

def wrap(x):
    w = len(keys)
    while x >= w:  x -= w
    while x <  0:  x += w
    return x

def get_pair(i):
    return img_pose_pairs[keys[wrap(i)]]

for i in range(len(keys)):
    pairs_to_use = []

    if KEEP_TURNTABLE:                 pairs_to_use.append(turntable_pair)
    for d in range(-IMGS_LEFT, 0):     pairs_to_use.append(get_pair(i + d))
    pairs_to_use.append(get_pair(i))
    for d in range(1, IMGS_RIGHT + 1): pairs_to_use.append(get_pair(i + d))

    x = 0
    output_img  = np.zeros((pairs_to_use[0][0].shape[0], sum([p[0].shape[1] for p in pairs_to_use]), pairs_to_use[0][0].shape[2]))
    output_pose = np.zeros(output_img.shape)
    for img, pose in pairs_to_use:
        assert img.shape[0] == output_img.shape[0], f"found mismatched image height when combining, found {img.shape[0]} != {output_img.shape[0]}"
        dx = img.shape[1]
        output_img[ :,x:x+dx] = img
        output_pose[:,x:x+dx] = pose
        x += dx
    assert x == output_img.shape[1], f"FATAL: expected shifted x to equal output width, found {x} != {output_img.shape[1]}"

    target_index = (1 if KEEP_TURNTABLE else 0) + IMGS_LEFT
    data = {
        "start": sum([p[0].shape[1] for p in pairs_to_use[:target_index  ]]),
        "end":   sum([p[0].shape[1] for p in pairs_to_use[:target_index+1]]),
    }

    folder = f"{next_folder}/{keys[i]}"
    if not os.path.exists(folder):  os.mkdir(folder)
    cv2.imwrite(f"{folder}/{INPUT_IMG_NAME}", output_img)
    cv2.imwrite(f"{folder}/{POSE_IMG_NAME}",  output_pose)
    with open(f"{folder}/{DATA_JSON_NAME}", "w") as f:
        f.write(json.dumps(data))

with open(f"{next_folder}/{TOUCH_FILE_NAME}", "w") as f:
    f.write("_")

