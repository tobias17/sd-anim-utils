# Stable Diffusion Animation Utilities

This repo contains a suite of utility programs written in Python to help create animations using Stable Diffusion

All of the programs read constants from settings.py - if anything is not where it expects it will let you know (most likely) and you either need to rename your files, or change a value in settings.py

## Getting Started

Install the required python packages
```
pip install opencv-python
pip install numpy
pip install rembg
```
If you want to run frame interpolation locally, you will also need some tensorflow packages. For this I used a conda env with python=3.11 and used these installs
```
pip install tensorflow
pip install tensorflow_hub
```

## Initializing a Workspace

To get a workspace started, you need to do these x things:
1. Create a folder in the workspaces directory, and change WORKSPACE in settings.py to that folder location
2. Create a poses folder in your workspace, and put all of your openpose pngs in there
3. Create a turntable.png pose picture, put it into the poses folder
4. Create a reference image following the turntable pose, and put it in the root of your workspace named source.png
5. Run `python generation.py` to initialize your first iteraction

After running step 5, if everything went well you should see a iter000 folder appear in your workspace. If you look inside, it should house a suite of folders, all named the same as all of the pose images in your poses folder, minus the turntable.

You can now run, folder by folder, the images and poses through stable diffusion to generate an initial version for all of your characters in each of their poses. Once you found an image you are satisfied with for a given pose, throw that image into the corresponding folder, alongside input.png, pose.png, and data.json (you should now have 4 files in that folder).

## Iterating a Workspace

To clean up the images and increase coherency, the iterate.py script will pull in surrounding images for the refinement process. For control over what gets pulled in, check out the KEEP_TURNTABLE, IMGS_LEFT, IMGS_RIGHT, and RANDOMIZE_ORDER variables at the top of settings.py.

Once you have the settings the way you want, run iterate.py. If everything went well, you should see an iter001 folder appear with all of your settings. If you do not like what you got, you can always change the settings, delete the iter folder, and run iterate.py again.

The process of running images through SD is exactly the same as before. Pull the images into your webui, run them through SD, and pull the result in the same folder the inputs came from.

To keep iterating, run iterate.py to generate a iter002, iter003, and so on folders to really refine your animation.

## Extracting and Cleaning

Once you are satisfied with the result, you can run extract.py. This will run through the latest iter folder, and pull out your subjects, placing them into and extracted folder in your workspace.

Running clean.py will take all of the images in the extracted folder, remove the background along with creating a variant with a clean background, and saving both copies into a cleaned folder.

## Interpolating (Optional)

I also included a script to interpolate between frames. This is a little more complex, required a more finicky python install library, along with source code changes to direct what animations to interpolate between.

Note: you can also use 3rd party interpolation. I have not used any of them, but RunwayML seems to have a decent one at first glance.

Heding down to Line 131, you will see 4 variables and a function.
1. ITERATIONS: how many iterations this program will run between frames. For every iteration, the number of frames will be doubled. For example if you had 4 animation frames, 1 iteration would create 4 new frames (8 total), 2 iterations would create 12 new frames (16 total), 3 iterations would create 28 new frames (32 total), etc.
2. LOOP_INTERP: whether the program will try to interpolate between the last and first frame. Set this to true if your animation is supposed to loop (like a run animation) and false if not (like an attack animation).
3. interp_folder: where it will scan for the files to interpolate between (default is the cleaned folder in your workspace). If you don't feel comfortable changing code, put the files you want to interpolate into a folder within cleaned and add that folder name to this variable.
4. to_interpolate: a function to determine if a file within interp_folder will be considered for animation. Default is return True while runs all files in that folder, but I left some examples for how one could limit files to certain animation segments.

Run iterpolate.py and you should see your interpolated frames getting generated.

## Spritesheet Generation

Once you have everything you want, you can run spritesheet.py to generate a spritesheet of your animation frames (without a background). There are a few settings within spritesheet.py to control some aspects of generation.
1. SKIP_EXISTING: the program automatically runs the remove background on images that do not have a rembg version. Setting this variable to True will have it run it for ALL images, even if one already exists. Mostly there for if you run interpolate a second time, and need the rembg version of iterpolated frame regenerated.
2. MAX_WIDTH: will create a second and so on row as the size of the spritesheet exceeds this value. Is there since Godot has a max image width of 16k for sprites.
3. RESIZE: by what factor images are resized when put into the spritesheet. Default is 1 (unchanged), but a value of 0.5 would downsize the image by 1/2 for both width and height (1/4 the pixels). This does NOT use AI, so I recommend avoiding upscaling using this feature.
