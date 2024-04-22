"""This file is designed to be run once at the start of analysis. It will take
the files in the data/train directories and move a given percentage of them to
the validation directory."""

import math
import os
import shutil

train_dir = "data/train"
validation_dir = "data/validation"

class_dirs = os.listdir(train_dir)

classes = [class_name for class_name in class_dirs if ".DS_Store" not in class_name]

for c in classes:
    dir_path = os.path.join(train_dir, c)
    files = os.listdir(dir_path)
    test_size = math.ceil(len(files) * 0.1)
    for f in files[:test_size]:
        shutil.move(os.path.join(dir_path, f), os.path.join(validation_dir, c))

