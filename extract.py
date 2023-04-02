##################################################
#                                                #
#   Extracts all of your images from the last    #
#   iteration that was run                       #
#                                                #
##################################################

from settings import *

import json
import cv2
import os

def folder_name(i):
    return f"{WORKSPACE}/{ITER_PREFIX}{i:03}"

curr_folder = ""
next_folder = ""
for i in range(1000):
    next_folder = folder_name(i)
    if not os.path.exists(next_folder):
        curr_folder = folder_name(i-1)
        break

print(f"extracting: {curr_folder}")
assert len(curr_folder) > 0, "could not find an iter folder, run generation.py to first create a workspace"

if not os.path.exists(EXTRACT_PATH):  os.mkdir(EXTRACT_PATH)

root_folders = [f for f in os.listdir(curr_folder) if f != TOUCH_FILE_NAME]
exclusion_list = [DATA_JSON_NAME, INPUT_IMG_NAME, POSE_IMG_NAME]

imgs = []

for root_folder in root_folders:
    root = f"{curr_folder}/{root_folder}"

    data_path = f"{root}/{DATA_JSON_NAME}"
    input_paths = [f for f in os.listdir(root) if f not in exclusion_list]
    assert len(input_paths) == 1, f"expected 1 extra file in {root_folder}, got {len(input_paths)}: {input_paths}"
    input_path = f"{root}/{input_paths[0]}"

    assert os.path.exists(data_path),  f"could not find data_path for {root_folder}, searched {data_path}"
    assert os.path.exists(input_path), f"could not find input_path for {root_folder}, searched {input_path}"

    with open(data_path) as f:
        data = json.loads(f.read())
    assert "start" in data, f"expected 'start' key in data json file for {root_folder} at {data_path}"
    assert "end" in data,   f"expected 'end' key in data json file for {root_folder} at {data_path}"

    input = cv2.imread(input_path)
    assert input is not None, f"input img was None for '{input_path}', expected value"
    assert input.shape[1] >= data["end"], f"input img must have width of at least data.json['end'], found {input.shape[1]} < {data['end']}"
    input = input[:,data["start"]:data["end"]]

    clean_img_path = f"{EXTRACT_PATH}/{root_folder}.png"
    cv2.imwrite(clean_img_path, input)
