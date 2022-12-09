import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage import filters
from skimage.filters import unsharp_mask
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import os
import pickle

# close all open figures
plt.close('all')

# relevant folder paths
label_path = "label_2000/"
raw_path = "raw_2000/"

n_comp = 100 # number of PCA components
n_img = 1000 # number of images to train on
mod = 'sav_edge' # image pre-processing

show_images = False # show images while pre-processing them
test_logreg = False # display a test image after training the model


logreg_pca_file = f'Final_Models/{n_img}_pca_{mod}.sav'
print(logreg_pca_file)


chunk_side = 16
n = int(1024/chunk_side)
chunk_per_image = int(n*n)

X = np.zeros((n_img*chunk_per_image, chunk_side*chunk_side))
Y = np.zeros((n_img*chunk_per_image,))
image = np.array([])

c = np.asarray(np.hsplit(np.asarray(np.hsplit(image, n)), n))
d = np.concatenate(c, axis=0)

# apply filter to the images and flatten them
for i,img in enumerate(os.listdir(label_path)):
    label = cv2.imread(label_path+img, 0)
    raw = cv2.imread(raw_path + img[:-3]+'png', 0)

    if mod == 'none':
        modified_img = raw
    
    elif mod == 'niblack':
        threshold = filters.threshold_niblack(raw)
        modified_img = (raw > threshold)

    elif mod == 'sauvola':
        threshold = filters.threshold_sauvola(raw)
        modified_img = (raw > threshold)

    elif mod == 'sav_edge':
        temp_img = unsharp_mask(raw, radius=5, amount=2)
        threshold = filters.threshold_sauvola(temp_img)
        modified_img = (temp_img > threshold)
    
    
    chunked_label = np.asarray(np.hsplit(np.asarray(np.hsplit(label, n)), n))
    flat_label = np.concatenate(chunked_label, axis=0)

    chunked_img = np.asarray(np.hsplit(np.asarray(np.hsplit(modified_img, n)), n))
    flat_chunked = np.concatenate(chunked_img, axis=0)

    for idx, ch in enumerate(flat_chunked):
        X[i*chunk_per_image+idx,:] = ch.reshape((1,ch.size)).astype(int)
        label_mode = np.argmax(np.bincount(flat_label[idx].flatten()))      
        Y[i*chunk_per_image+idx] = label_mode

    if show_images:
        plt.figure()
        plt.imshow(modified_img, cmap='gray')
        plt.xlabel('after filter has been applied')

        plt.figure()
        plt.imshow(raw, cmap='gray')
        plt.xlabel('raw image')

        plt.figure()
        l = len(flat_chunked)
        plt.imshow(Y[i*l:(i+1)*l].reshape((n,n)), cmap='gray')
        plt.xlabel('chunked labeled Y')

        plt.figure()
        plt.imshow(label, cmap='gray')
        plt.xlabel("labeled y")

        plt.show()

    # break after training on n_img images
    if (i == n_img-1):
        break

# Apply PCA to X matrix
pca = PCA(n_components=n_comp)
pca.fit(X)
PCA_X = pca.transform(X).astype(int)

# Create the Logistic Regression Model
logreg_pca = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l1', class_weight='balanced')

# Train the logreg
logreg_pca.fit(PCA_X, Y)
pickle.dump(logreg_pca, open(logreg_pca_file, 'wb'))

print('Done training logreg model')

if test_logreg:
    raw_path = "C:/Users/visha/Documents/Stanford/Fall_2022/CS_229/Project/ai4mars-dataset-merged-0.1/ai4mars-dataset-merged-0.1/msl/images/edr/NLB_527820640EDR_F0580000NCAM00253M1.JPG"
    raw_img = cv2.imread(raw_path, 0)

    labeled_image_path = "C:/Users/visha/Documents/Stanford/Fall_2022/CS_229/Project/ai4mars-dataset-merged-0.1/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree/NLB_527820640EDR_F0580000NCAM00253M1_merged.png"
    labeled_image = cv2.imread(labeled_image_path, 0)

    loaded_pca_model = pickle.load(open(logreg_pca_file, 'rb'))

    chunked_train = np.asarray(np.hsplit(np.asarray(np.hsplit(raw_img, n)), n))
    flat_train = np.concatenate(chunked_train, axis=0)
    TRAIN_X = np.zeros((chunk_per_image,chunk_side**2))
    for idx,i in enumerate(flat_train):
        TRAIN_X[idx,:] = i.flatten()

    pca = PCA(n_components=150)
    pca_pred = loaded_pca_model.predict(pca.fit_transform(TRAIN_X))

    labeled_image = np.array([min(255, 50*i) for i in labeled_image.flatten()])
    pca_pred = np.array([min(255, 50*i) for i in pca_pred.flatten()])

    plt.figure()
    plt.imshow(raw_img, cmap="gray")
    plt.title("Raw image")

    plt.figure()
    plt.imshow(labeled_image.reshape((1024, 1024)), cmap="gray")
    plt.title("Labeled image")

    plt.figure()
    plt.imshow(pca_pred.reshape((n,n)), cmap="gray")
    plt.title("PCA prediction")

    plt.show()
