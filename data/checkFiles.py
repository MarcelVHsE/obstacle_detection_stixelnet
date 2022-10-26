#!/usr/bin/python

import numpy as np
import os


files = ["waymo_train.txt"]#, "waymo_val.txt"]
minX = 0
maxX = 1919

minY = 0
maxY = 1279

imgPrefix = "waymo_stixel_images"

count = 0
countInvalid = 0

for file in files:
    print(f"Processing %s" % (file))
    fileData = np.genfromtxt(file, dtype=None , delimiter=" ")

    for item in fileData:        
        fileName = item[0].decode("utf-8")
        xCoord   = item[1]
        yCoord   = item[2]

        valid = True
        if not os.path.exists(os.path.join(imgPrefix, fileName)):
            print(f"{fileName} does not exist")
            valid = False

        if xCoord < minX or xCoord > maxX:
            print(f"x {xCoord} out of range in {fileName}")
            valid = False

        if yCoord < minY or yCoord > maxY:
            print(f"y {yCoord} out of range in {fileName}")
            valid = False

        if not valid:
            countInvalid = countInvalid + 1


print(f"{countInvalid} / {count} files are invalid")

