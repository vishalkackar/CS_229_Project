"""
Generate a score (dice coefficient) that tells you how accurate a prediction label was
compared to the actual labels generated for an image
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


true_labels_directory = "true_labels_for_scoring/" # filepath to folder containing the given labels
generated_labels_directory = "generated_labels_for_scoring/" #filepath to folder containing the labels we generated with logreg/CNN and want to score
raw_images_directory = "Logreg_Test_Images/"
model_directory = "Final_Models/"



def generate_dice_coefficients(generated_labels, true_labels, mask_only=True, classes=['soil', 'bedrock', 'sand', 'big_rock', 'no_label']):
    """
    Score each generated label in generated_labels by comparing it to
    the corresponding true image label in true_labels

    Kwargs:
        generated_labels: list of labels (each label is a numpy array) generated through logreg or CNN
        true_labels: list of true labels (each label is a numpy array) provided by AI4Mars
        classes: list of classes we're checking accuracy for
    
    Returns:
        A tuple of...
        1) The mean of the dice scores for all of the image labels
        2) A numpy array of scores between 0 and 1 corresponding to the generated label
        accuracy for each image
        
    """
    print(classes)
    if mask_only:
        classes = ['any_terrain', 'no_label']
        class_dict = {'any_terrain': 0, 'no_label': 255}
    else:
        class_dict = {'soil': 0, 'bedrock': 50, 'sand': 100, 'big_rock': 150, 'no_label': 255}
    
    dice_coefficients = np.zeros(len(generated_labels)) # array containing the dice coefficients corresponding to each generated label
    used_classes = [] # list containing the grayscale values for the classes we're using
    for i in classes:
        used_classes.append(class_dict[i])
    
    for idx, gen_array in enumerate(generated_labels):
        true_array = true_labels[idx]
        true_array[true_array < 255] *= 50
        intersections = np.zeros(len(used_classes))
        if mask_only:
            inverted_intersections = np.zeros(len(used_classes))
        individual_list_dice = []

        # generate accuracy for each individual class
        
        for val_idx, val in enumerate(used_classes):
            # create a truth array for the true and generated labels. Truth
            # arrays have a 1 where the value matches the class we are 
            # checking the accuracy for and a 0 otherwise
            if mask_only:
                gen_array[gen_array!=255] = 0
                true_array[true_array!=255] = 0

                # Logreg masks well, but sometimes inverts the color of the terrain/masked area. To deal with these cases, 
                # we make inverted arrays as well and pick which one (inverted or original) has the highest dice score.
                # Note that we only used this when we are scoring for a mask, and NOT when we score for accuracy across
                # all terrain types individually (because the mask can be a random color potentially)
                inverted_gen_array = np.array(gen_array)
                inverted_gen_array[gen_array==0] = 1
                inverted_gen_array[gen_array==1] = 0

                inverted_true_array = np.array(true_array)
                inverted_true_array[true_array==1] = 0
                inverted_true_array[true_array==0] = 1
            
            truth_array_gen = np.array(gen_array)
            # truth_array_gen[truth_array_gen!=val]
            truth_array_gen[gen_array==val] = 1
            truth_array_gen[gen_array!=val] = 0

            truth_array_true = np.array(true_array)
            truth_array_true[true_array==val] = 1
            truth_array_true[true_array!=val] = 0

            if mask_only:
                inverted_truth_array_gen = np.array(inverted_gen_array)
                # truth_array_gen[truth_array_gen!=val]
                inverted_truth_array_gen[inverted_gen_array==val] = 1
                inverted_truth_array_gen[inverted_gen_array!=val] = 0

                inverted_truth_array_true = np.array(inverted_true_array)
                inverted_truth_array_true[inverted_true_array==val] = 1
                inverted_truth_array_true[inverted_true_array!=val] = 0
                
            
            # create inverted labels so we can use a logical and on this to see where the truth and prediction agree on pixels where the pixel
            # is not labelled as this class. Note that this inversion is different from the mask only inversion in the mask-only case
            inverted_true = np.logical_not(truth_array_true)
            inverted_gen = np.logical_not(truth_array_gen)
            
            # do a logical and to find where the generated label matches the true label for this class for all pixels 
            # (i.e. find the number of pixels that both agree is either part of or not part of the current label)
            intersection = np.sum(np.logical_and(truth_array_true, truth_array_gen) + np.logical_and(inverted_true, inverted_gen))
            intersections[val_idx] = intersection # save the number of correct pixels for this class

            if mask_only:
                inverted_inverted_true = np.logical_not(inverted_truth_array_true)
                inverted_inverted_gen = np.logical_not(inverted_truth_array_gen)
                
                # do a logical and to find where the generated label matches the true label for this class for all pixels 
                # (i.e. find the number of pixels that both agree is either part of or not part of the current label)
                inverted_intersection = np.sum(np.logical_and(inverted_truth_array_true, inverted_truth_array_gen) + np.logical_and(inverted_inverted_true, inverted_inverted_gen))
                inverted_intersections[val_idx] = inverted_intersection # save the number of correct pixels for this class

                # Pick the mask (original or inverted color) that is best.
                individual_list_dice.append(max(intersection, inverted_intersection))
                # individual_list_dice.append(intersection)

            
            else:
                individual_list_dice.append(intersection)

        # generate the dice coefficient for one image label
        dice = 2*(np.sum(intersections)/len(used_classes))/(np.size(gen_array) + np.size(true_array))
        dice_coefficients[idx] = dice
    
    return(np.mean(dice_coefficients), dice_coefficients)


def import_images(directory, desired_image_size=1024):
    """
    Import all of the label files found in the provided directory as numpy arrays

    Kwargs:
        directory: filepath to directory containing desired images
        desired_image_size: desired size (in pixels) of the images

    Returns:
        A list of numpy arrays corresponding to each image
    """

    image_list = []
    name_list = []
    for i, img in enumerate(os.listdir(directory)):
        if img != '.DS_Store':
                image_array = cv2.imread(directory + img, 0)

                # if the image size is wrong, scale it up to the correct one
                if image_array.shape[0] != desired_image_size:
                    image_array = cv2.resize(image_array, (desired_image_size, desired_image_size), interpolation=cv2.INTER_NEAREST)
                # plt.imshow(image_array, cmap='gray')
                # plt.show()
        name_list.append(img)
        image_list.append(image_array)
    return (image_list, name_list)


def import_models(model_directory):
    """
    Load a list of models for logistic regression predictions

    Kwargs:
        model_directory: directory containing pretrained model files
    
    Returns:
        model_list: A dictionary mapping loaded models to the filename of the model file
    """
    models = {}
    os.chdir('Final_Models/')
    # print(os.getcwd())
    for idx, model in enumerate(os.listdir('.')):
        # print("hello my baby hello my darling hello my ragtime gaaaaal")
        # print(model)
        if model != '.DS_Store':
            # print("hello my baby hello my darling hello my ragtime gaaaaal")
            # print(model)
            # print("oeiurbiouwbf")
            # print(model)
            loaded_model = pickle.load(open(model, 'rb'))
            models[model] = loaded_model
    os.chdir('..')
    return models



def generate_prediction_labels(model, raw_images, model_name, image_names):
    """
    Generate prediction labels for one model for each imaeg

    Kwargs:
        model: A pretrained model for LogisticRegression
        raw_images: A list of raw images to generate predictions on
    
    Returns:
        A list of numpy arrays. Each numpy array is a prediction label
    """
    # model_objects = list(models.keys())
    # pca = PCA(n_components=75) # for models trained on 1500 images
    pca = PCA(n_components=100) # for models trained on 1000 images
    # pca = PCA(n_components=150) # for models trained on less than 1000 images
    prediction_list = []
    save_folder = 'gen_labels_' + model_name + '/'
    print(f"Generating results for {model_name}")
    # print(len(image_names))

    chunk_dimension = 16
    n = int(1024/chunk_dimension)
    chunk_per_image = n**2

    for idx, image in enumerate(raw_images):
        # print(image.shape)

        chunked_train = np.asarray(np.hsplit(np.asarray(np.hsplit(image, n)), n))
        flat_train = np.concatenate(chunked_train, axis=0)
        TRAIN_X = np.zeros((chunk_per_image, chunk_dimension**2))

        for idy, i in enumerate(flat_train):
            TRAIN_X[idy, :] = i.flatten()

        # print(idx)
        pca_pred = model.predict(pca.fit_transform(TRAIN_X))
        pca_pred = pca_pred.reshape((n,n))
        pca_pred = cv2.resize(pca_pred, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # print("My prediction")
        # print(pca_pred.shape)

        cv2.imwrite(save_folder + image_names[idx], pca_pred)
        prediction_list.append(pca_pred)
        # print(save_folder + image_names[idx])
        print("Predicted a label")
    
    return prediction_list


def main():
    generate_labels = False
    desired_labels_directory = 'gen_labels_1500_pca_niblack/' # if you already generated labels, you can pick a folder containing predictions and only score that
    desired_image_size = 1024
    # classes = ['No label']
    models = import_models(model_directory)
    # print("THESE ARE THE MODELS")
    # print(models)
    (raw_images, image_names) = import_images(raw_images_directory)
    (true_labels, _) = import_images(true_labels_directory) # list of numpy arrays, each array corresponds to a true label for an image
    # generated_labels = import_images(generated_labels_directory, desired_image_size) # list of numpy arrays, each array corresponds to a generated label
    results_dictionary = {}

    if generate_labels:
        for model in models:
            model_name = os.path.splitext(model)[0]
            generated_labels = generate_prediction_labels(models[model], raw_images, model_name, image_names)
            (mean_score, dice_scores) = generate_dice_coefficients(generated_labels, true_labels, mask_only=True)
            results_dictionary[model_name] = (mean_score, dice_scores)
            print(f"Mean dice score for {model_name}: {mean_score}")

    else:
        (generated_labels, _) = import_images(desired_labels_directory)
        print("oringioenbgoiwnfoiwebofibweoifbwofbweoiufbwef")
        print(generated_labels[0].shape)
        (mean_score, dice_scores) = generate_dice_coefficients(generated_labels, true_labels, mask_only=False)
        results_dictionary[desired_labels_directory] = (mean_score, dice_scores)

    for result in results_dictionary:
        dice_score = results_dictionary[result][0]
        individual_image_dice_scores = results_dictionary[result][1]
        # print(f"Mean accuracy: {mean_score}")
        # Note: dice_score is the mean dice score for a given model over all image predictions.
        # individual_image_dice_scores is the list of the dice scores for a given model for all image predictions
        print(f"Mean dice score for {result}: {dice_score}")

if __name__=='__main__':
    main()