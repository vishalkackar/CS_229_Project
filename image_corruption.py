import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.util import view_as_blocks
import random


# raw images
raw_path = "Logreg_Test_Images/"
masked_path = "test_selected/"
save_directory_raw = "final_gaussian_clump_dead_damage/"
save_directory_masked = 'masked_' + save_directory_raw



def create_saturated_lines(img, masked_img, n_rows, n_cols, saturation_val=255, masked = True):
    """
    Randomly adds in n_rows of saturated pixel rows, n_cols of random saturated columns to a raw image
    
    kwargs:
        img: image array imported from openCV
        n_rows: number of rows to corrupt
        n_cols: number of columns to corrupt
        saturation_val: grayscale value to set corrupted pixels to. Set to 255 (white) by default
    """
    assert(n_rows < img.shape[0])
    assert(n_cols < img.shape[1])

    idx_array = np.arange(img.shape[0])
    row_idxs = np.random.choice(idx_array, n_rows, replace=False)
    col_idxs = np.random.choice(idx_array, n_cols, replace=False)
    # if masked:
    original_img = np.array(masked_img)
    masked_img[row_idxs, :] = saturation_val
    masked_img[:, col_idxs] = saturation_val 
    masked_img[original_img==255] = 255
    # else:
    img[row_idxs, :] = saturation_val
    img[:,col_idxs] = saturation_val

    return (img, masked_img)

def create_dead_pixel_lines(img, masked_img, n_rows=10, n_cols=10, masked= True):
    """
    Randomly add in n_rows of saturated pixel rows, n_cols of random saturated columns to a raw image
    
    kwargs:
        img: image array imported from openCV
        n_rows: number of rows to corrupt
        n_cols: number of columns to corrupt
    """
    assert(n_rows < img.shape[0])
    assert(n_cols < img.shape[1])

    idx_array = np.arange(img.shape[0])
    row_idxs = np.random.choice(idx_array, n_rows, replace=False)
    col_idxs = np.random.choice(idx_array, n_cols, replace=False)
    # if masked:
    original_img = np.array(masked_img)
    masked_img[row_idxs, : ] = 0
    masked_img[:, col_idxs] = 0 
    masked_img[original_img==255] = 255

    # else:
    img[row_idxs, :] = 0
    img[:, col_idxs] = 0
    
    return (img, masked_img)

def create_random_dead_pixels(img, masked_img, n_pixels=10000, distribution='gaussian', mean = 512, std = 45):
    """
    Create a random cluster of n_pixels of dead pixels distributed with mean and std in
    a Gaussian or uniform distribution.

    kwargs:
        img: raw image array imported from openCV
        masked_img: masked image array imported from openCV
        n_pixels: number of corrupted pixels to create
        distribution: distribution used for pixel distribution (gaussian and uniform implemented so far)
        saturation_val: grayscale value to set corrupted pixels to. White (255) by default
        mean: center from which we want to create a cluster of dead pixels for gaussian 
        std: standard deviation of the cluster of dead pixels for gaussian
    """
    assert(mean <= 1023)
    assert(n_pixels <= 1023**2)
    


    if distribution == 'uniform':
        pixel_array_x_vals_float = np.rint(np.random.uniform(0,1023,n_pixels))
        pixel_array_y_vals_float = np.rint(np.random.uniform(0,1023,n_pixels))
        pixel_array_x_vals = pixel_array_x_vals_float.astype(np.int32)
        pixel_array_y_vals = pixel_array_y_vals_float.astype(np.int32)

    elif distribution == 'gaussian':
        pixel_array_x_vals = np.zeros(n_pixels, dtype=int)
        pixel_array_y_vals = np.zeros(n_pixels, dtype=int)
        # uncomment next two lines for tight cluster of pixels
        # note: used std = 45 for clumps
        mean1 = random.randint(0, 1023)
        mean2 = random.randint(0, 1023)
    
        for i in range(n_pixels):
            # pixel_array_x_vals[i] = int(round(np.random.normal(mean, std)))
            # pixel_array_y_vals[i] = int(round(np.random.normal(mean, std)))

            # uncomment next two lines for tight cluster of pixels 
            pixel_array_x_vals[i] = int(round(np.random.normal(mean1, std)))
            pixel_array_y_vals[i] = int(round(np.random.normal(mean2, std)))

            # keep pixels within the bounds; this rarely happens, here just in case. Doesn't cause clustering along edges
            # for the most part. 
            if pixel_array_x_vals[i] > 1023:
                pixel_array_x_vals[i] = 1023
            if pixel_array_y_vals[i] > 1023:
                pixel_array_y_vals[i] = 1023
            if pixel_array_x_vals[i] < 0:
                pixel_array_x_vals[i] = 0
            if pixel_array_y_vals[i] < 0:
                pixel_array_y_vals[i] = 0
    
    # if masked:
    original_img = np.array(masked_img) # make a copy of the original image array
    masked_img[pixel_array_x_vals, pixel_array_y_vals] = 0 # set appropriate pixels to dead pixels
    masked_img[original_img > 250] = 255 # reset any pixels that were part of the mask 

    # else:
    img[pixel_array_x_vals, pixel_array_y_vals] = 0
    
    return (img, masked_img)

def create_random_saturated_pixels(img, masked_img, n_pixels=10000, distribution='uniform', saturation_val=255, mean = 512, std = 300):
    """
    Create a random cluster of n_pixels of saturated pixels distributed with mean and std in
    a Gaussian or uniform distribution. 

    kwargs:
        img: raw image array imported from openCV
        masked_img: masked image array imported from openCV
        n_pixels: number of corrupted pixels to create
        distribution: distribution used for pixel distribution (gaussian and uniform implemented so far)
        saturation_val: grayscale value to set corrupted pixels to. White (255) by default
        mean: center from which we want to create a cluster of dead pixels for gaussian 
        std: standard deviation of the cluster of dead pixels for gaussian
    """
    assert(mean <= 1023)
    assert(n_pixels <= 1023**2)

    if distribution == 'uniform':
        pixel_array_x_vals_float = np.rint(np.random.uniform(0,1023,n_pixels))
        pixel_array_y_vals_float = np.rint(np.random.uniform(0,1023,n_pixels))
        pixel_array_x_vals = pixel_array_x_vals_float.astype(np.int32)
        pixel_array_y_vals = pixel_array_y_vals_float.astype(np.int32)

    elif distribution == 'gaussian':
        pixel_array_x_vals = np.zeros(n_pixels, dtype=int)
        pixel_array_y_vals = np.zeros(n_pixels, dtype=int)
    
        for i in range(n_pixels):
            pixel_array_x_vals[i] = int(round(np.random.normal(mean, std)))
            pixel_array_y_vals[i] = int(round(np.random.normal(mean, std)))

            # will fix this later, just a quick patch
            if pixel_array_x_vals[i] > 1023:
                pixel_array_x_vals[i] = 1023
            if pixel_array_y_vals[i] > 1023:
                pixel_array_y_vals[i] = 1023
            if pixel_array_x_vals[i] < 0:
                pixel_array_x_vals[i] = 0
            if pixel_array_y_vals[i] < 0:
                pixel_array_y_vals[i] = 0
    
    # if masked:
    original_img = np.array(masked_img)
    masked_img[pixel_array_x_vals, pixel_array_y_vals] = saturation_val
    masked_img[original_img == 255] = 255

    # else:
    img[pixel_array_x_vals, pixel_array_y_vals] = saturation_val

    return (img, masked_img)


raw_path_files = os.listdir(raw_path)
masked_path_files = os.listdir(masked_path)
masked_path_files.remove('.DS_Store')
for i,img in enumerate(masked_path_files):
    print('BERRIES AND CREAM')
    if img != '.DS_Store':
        print(img)
        raw = cv2.imread(raw_path + raw_path_files[i], 0)
        masked = cv2.imread(masked_path + img, 0)
        # print(raw.shape)
        # (corrupt_image_raw, corrupt_image_masked) = create_random_saturated_pixels(raw, masked, 260000)
        (corrupt_image_raw, corrupt_image_masked) = create_random_dead_pixels(raw, masked, 50000)
        # (corrupt_image_raw, corrupt_image_masked) = create_dead_pixel_lines(raw, masked, 130, 130)
        # (corrupt_image_raw, corrupt_image_masked) = create_saturated_lines(raw, masked, 130, 130)



        # print(corrupt_image)
        # plt.figure(i)
        # plt.imshow(corrupt_image, cmap='gray')
        # plt.show()
        # plt.savefig(save_directory + 'damaged_' + img)
        cv2.imwrite(save_directory_raw + 'damaged_' + img, corrupt_image_raw)
        cv2.imwrite(save_directory_masked + 'damaged_' + img, corrupt_image_masked)
        # plt.close()
        

   
# filename = 'test_damaged_image.png'
# plt.figure()
# plt.imshow(corrupt_image, cmap='gray')
# plt.savefig(save_directory + filename)

# plt.show()