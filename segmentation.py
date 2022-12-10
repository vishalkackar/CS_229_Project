import skimage
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
from skimage.io import imsave

data_directory = "raw_prediction_data/"
save_directory = "sauvola_prediction_data/"

for filename in os.listdir(data_directory):
    if filename.endswith('.JPG'):
        print(filename)
        image = skimage.io.imread(data_directory + filename)

        # three different types of threshold
        # threshold = filters.threshold_otsu(image)
        # threshold = filters.threshold_niblack(image)
        threshold = filters.threshold_sauvola(image)

        binarized_image = (image > threshold)
        print("shape")
        print(binarized_image.shape)
        plt.imshow(binarized_image, cmap='gray')
        skimage.io.imsave(save_directory + filename, binarized_image)
        # plt.savefig(save_directory + filename)
        # plt.savefig(save_directory  + 'sauvola_' + filename)
        # plt.show()
