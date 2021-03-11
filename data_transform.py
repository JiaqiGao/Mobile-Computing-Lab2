import os
from PIL import Image, ImageFilter

# Prerequisite:
# Create an empty copy of the class dataset folder and rename it the path name you chose below (ex: "transformed_class_dataset_4")

# get all image filenames
og_path = "class_dataset"
new_path = "transformed_class_dataset_4"
data_ids = ['thumb_down', 'swiping_up', 'swiping_down', 'thumb_up', 'no_gesture', 'swiping_left', 'shaking_hand', 'swiping_right', 'doing_other_things', 'stop_sign']

def get_frame_names(path, id):
    filenames = os.listdir(path+"/"+id+"/frames")
    return [path+"/"+id+"/frames/"+x for x in filenames]

def new_transformed_filepath(new_path, og_filepath):
    return new_path+"/"+og_filepath.split('/',1)[-1]

def apply_blur_transform_and_save(filepath):
    im = Image.open(filepath)
    im1 = im.filter(ImageFilter.BoxBlur(10))
    im1.save(new_transformed_filepath(new_path, filepath))

for i in data_ids:
    for f in get_frame_names(og_path,i):
        apply_blur_transform_and_save(f)