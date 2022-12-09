# CS_229_Project
Fall 2023 CS 229 Project - Ryan Chen, Dylan Goff, Vishal Kackar

Deep and Supervised Learning Approaches to Terrain Classification on Images From the Mars Rover

Folders:\
CNN Model - Files used to train the CNN\
Logreg_Models - Trained logistic models\
logreg_pca.py - File used to train the logistic regression models\\

logreg_pca.py:\
-To train a model first specify the raw and labeled image paths on the local machine. The raw path corresponds to the training images and the labeled path corresponds to the training labels for each image in the raw path.\\

-Next, specify the number of images to train on, keeping in mind that the memory required to train the model increases as the number of images increases.\\

-Specify the number of PCA components to keep for each image. This may have to be lowered depending on how many images are selected for training.\\

-Finally, select the modifications to apply to the image:\
&nbsp;&nbsp;&nbsp;&nbsp;none: only PCA\
&nbsp;&nbsp;&nbsp;&nbsp;niblack: niblack thresholding\
&nbsp;&nbsp;&nbsp;&nbsp;sauvola: sauvola thresholding\
&nbsp;&nbsp;&nbsp;&nbsp;sav_edge: unsharp masking followed by sauvola thresholding
