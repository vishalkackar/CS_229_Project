import cv2
import os
import numpy as np
import pprint as pp
import pandas as pd

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
###############

def matrix_gen(folder_path, filename):
	f = os.path.join(folder_path, filename)
	img_mat = cv2.imread(f, 0)
	return img_mat

def compress(matrix, factor):
	mat_comp = matrix[::factor,::factor]
	return mat_comp

def perform_comp(full_img_path, comp_path):
	for i, filename in enumerate(os.listdir(full_img_path)):
		if filename[0:1] == "N":
			print(i)
			image_arr = matrix_gen(full_img_path, filename)
			comp_image_arr = compress(image_arr, 8)
			cv2.imwrite(comp_path+ filename, comp_image_arr)

def main():
	perform_comp(path_masked,path_maskedcomp)
	perform_comp(path_test_min1, path_test_min1_comp)
	perform_comp(path_test_min2, path_test_min2_comp)
	perform_comp(path_test_min3, path_test_min3_comp)
	perform_comp(path_train, path_traincomp)

if __name__ == "__main__":
    main()