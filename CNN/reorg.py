import cv2
import os
import numpy as np
import pprint as pp
import pandas as pd
import shutil

#####PATHS#####
path_mxy = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/images/mxy/'
path_rng = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/images/rng-30m/'
path_edr = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/images/edr/'
path_train = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/train/'
path_test_min1 = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/test/masked-gold-min1-100agree/'
path_test_min2 = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/test/masked-gold-min2-100agree/'
path_test_min3 = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/test/masked-gold-min3-100agree/'
path_masked = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/images/masked/'
path_maskedcomp = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/images/maskedcomp/'
path_test_min1_comp = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/test/masked-min1-comp/'
path_test_min2_comp = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/test/masked-min2-comp/'
path_test_min3_comp = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/test/masked-min3-comp/'
path_traincomp = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/msl/labels/traincomp/'
path_images = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/model/images/'
path_masks = '/Users/shirleychen/Documents/Fall2022/CS229/dataset/model/masks/'
###############

def findmove(src, dst, test):
	for i, filename in enumerate(os.listdir(src)):
		print(i)
		if filename[0:1] == "N":
			if test == 1:
				filename_msk = filename[0:13] + "MSK" + filename[16:-11] + ".JPG"
			if test == 0:
				filename_msk = filename[0:13] + "MSK" + filename[16:-3] + "JPG"
			newname = filename[0:13] + ".tiff"
			f = os.path.join(path_masked, filename_msk)
			t = os.path.join(dst, newname)
			shutil.copy(f, t)

def labelmove(src, dst):
	for i, filename in enumerate(os.listdir(src)):
		print(i)
		if filename[0:1] == "N":
			newname = filename[0:13] + ".tiff"
			f = os.path.join(src, filename)
			t = os.path.join(dst, newname)
			shutil.copy(f, t)

def main():
	# findmove(path_train, path_images, 0)
	# findmove(path_test_min2_comp, path_images, 1)
	labelmove(path_train, path_masks)
	labelmove(path_test_min2, path_masks)


if __name__ == "__main__":
    main()