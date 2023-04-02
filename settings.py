WORKSPACE = "workspaces/001"

# VRAM sensitive constants
KEEP_TURNTABLE = True
IMGS_LEFT  = 0
IMGS_RIGHT = 1
RANDOMIZE_ORDER = False

BG_COLOR = (127, 127, 127) # in BlueGreenRed, not RedGreenBlue!

ITER_PREFIX = "iter"
DATA_NAME = "data.json"

POSE_IMG_NAME = "pose.png"
INPUT_IMG_NAME = "input.png"
DATA_JSON_NAME = "data.json"
TOUCH_FILE_NAME = ".created"

SOURCE_NAME = "source.png"
SOURCE_PATH = f"{WORKSPACE}/{SOURCE_NAME}"
SOURCE_S_NAME = "source_s.png"
SOURCE_S_PATH = f"{WORKSPACE}/{SOURCE_S_NAME}"
SOURCE_G_NAME = "source_g.png"
SOURCE_G_PATH = f"{WORKSPACE}/{SOURCE_G_NAME}"

POSES_NAME     = "poses"
POSES_FOLDER   = f"{WORKSPACE}/{POSES_NAME}"
TURNTABLE_NAME = "turntable.png"
TURNTABLE_PATH = f"{POSES_FOLDER}/{TURNTABLE_NAME}"

EXTRACT_FOLDER = "extracted"
EXTRACT_PATH   = f"{WORKSPACE}/{EXTRACT_FOLDER}"
CLEAN_FOLDER = "cleaned"
CLEAN_PATH   = f"{WORKSPACE}/{CLEAN_FOLDER}"
SHEET_NAME   = "sheet.png"
SHEET_PATH   = f"{CLEAN_PATH}/{SHEET_NAME}"