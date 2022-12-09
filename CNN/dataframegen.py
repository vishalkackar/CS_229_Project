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

def dfmaker(folder_path, save_name, test):
	label_df = pd.DataFrame(columns=['keyname','maskedimg','labels'])
	for i, filename_edr in enumerate(os.listdir(folder_path)):
		print(i)
		if test == 1:
			filename_msk = filename_edr[0:13] + "MSK" + filename_edr[16:-11] + ".JPG"
		if test == 0:
			filename_msk = filename_edr[0:13] + "MSK" + filename_edr[16:-3] + "JPG"
		mask_image_arr = matrix_gen(path_maskedcomp, filename_msk)
		label_arr = matrix_gen(folder_path, filename_edr)
		newentry = pd.DataFrame([[filename_edr, mask_image_arr, label_arr]], columns=['keyname','maskedimg','labels'])
		label_df = pd.concat([label_df,newentry])
	label_df.to_pickle(save_name) 
	return(label_df)

def main():
	df_train = dfmaker(path_train, "./traindf.pkl", 0)
	df_test_min1 = dfmaker(path_test_min1_comp, "./testmin1df.pkl", 1)
	df_test_min2 = dfmaker(path_test_min2_comp, "./testmin2df.pkl", 1)
	df_test_min3 = dfmaker(path_test_min3_comp, "./testmin3df.pkl", 1)


if __name__ == "__main__":
    main()