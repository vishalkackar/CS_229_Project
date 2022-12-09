import cv2
import os
import numpy as np
import pprint as pp

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

def maskup(image_arr, mask_arr):
	masked_img_arr = np.ma.array(image_arr, mask = mask_arr)
	return masked_img_arr.filled(fill_value=255)

def main():
	# total = 18137
	for i, filename_edr in enumerate(os.listdir(path_edr)):
		if filename_edr[0:1] == "N":
			filename_mxy = filename_edr[0:13] + "MXY" + filename_edr[16:-3] + "png"
			filename_rng = filename_edr[0:13] + "RNG" + filename_edr[16:-3] + "png"
			mars_image_arr = matrix_gen(path_edr, filename_edr)
			mxy_mask_arr = matrix_gen(path_mxy, filename_mxy)
			rng_mask_arr = matrix_gen(path_rng, filename_rng)
			masked_img_arr = maskup(mars_image_arr, mxy_mask_arr)
			double_masked_img_arr = maskup(masked_img_arr, rng_mask_arr)
			filename_masked = filename_edr[0:13] + "MSK" + filename_edr[16:-3] + "JPG"
			cv2.imwrite(path_masked + filename_masked, double_masked_img_arr)

if __name__ == "__main__":
    main()
    