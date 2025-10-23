# Lesion-Segmention-and-Regresssion-Analysis
## Project Directory Structure

Here’s what your project folder should look like. This layout organizes everything logically.

/BraTS_Segmentation/
|
|-- data/
|   |-- train/
|   |-- validation/
|   |-- test/
|
|-- saved_models/
|
|-- config.py
|-- dataset.py
|-- model.py
|-- metrics.py
|-- losses.py
|-- train.py
|-- evaluate.py
|-- predict.py
|-- utils.py
|-- requirements.txt

## File-by-File Breakdown

Here’s what each Python (.py) file will do.

## config.py

This file is for all your settings and hyperparameters. Keeping them here means you don't have to hard-code them inside your training or model scripts.

    Purpose: Central configuration hub.

    What it will contain:

        Paths: DATA_DIR, SAVED_MODEL_PATH, etc.

        Hyperparameters: LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS.

        Model Specs: IN_CHANNELS (e.g., 3 for T1, T2, Flair), NUM_CLASSES (e.g., 1 for the lesion mask).

        Data Specs: IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH.

## dataset.py

This script handles loading your .nii files and preparing them for the model.

    Purpose: Data loading and preprocessing.

    What it will contain:

        A custom BraTSDataset class that inherits from torch.utils.data.Dataset.

        In the __getitem__ method, it will:

            Load the T1, T2, and Flair .nii files for a single patient using a library like nibabel.

            Load the corresponding ground truth segmentation mask.

            Stack the T1, T2, and Flair scans together to create a multi-channel 3D volume (e.g., shape [3, 128, 128, 128]).

            Apply data augmentations and preprocessing (e.g., normalization, resizing, random flips).

            Return the processed 3D image tensor and the corresponding mask tensor.

## model.py

This is where you'll define the architecture of your 3D U-Net. 🧠

    Purpose: Defines the neural network.

    What it will contain:

        A UNet3D class that inherits from torch.nn.Module.

        Helper modules like a DoubleConv block (Conv3D -> BatchNorm3D -> ReLU -> Conv3D -> BatchNorm3D -> ReLU).

        The Encoder (downsampling path) using DoubleConv blocks and MaxPool3d.

        The Decoder (upsampling path) using ConvTranspose3d and DoubleConv blocks.

        Logic for the skip connections that concatenate feature maps from the encoder to the decoder.

        A final 1x1x1 convolution to produce the output segmentation map.

## metrics.py

This file contains functions to calculate all your performance scores.

    Purpose: Evaluation metrics.

    What it will contain:

        A function dice_score(preds, targets) that calculates the Dice Similarity Coefficient.

        A function iou_score(preds, targets) for Intersection over Union (Jaccard Index).

        Other relevant functions like sensitivity() and specificity() if needed.

## losses.py

While you can use built-in losses, segmentation tasks often benefit from custom loss functions.

    Purpose: Define loss functions suitable for segmentation.

    What it will contain:

        A DiceLoss class, as it works very well for imbalanced datasets (where the lesion is a tiny part of the image).

        A combination loss, like TverskyLoss or a mix of DiceLoss and BinaryCrossEntropy.

## train.py

This is the main script you'll run to train your model.

    Purpose: The training and validation engine.

    What it will contain:

        Logic to import settings from config.py.

        Functions to create your model, loss function (from losses.py), and optimizer (e.g., Adam).

        Code to set up the DataLoader for both training and validation sets using your BraTSDataset.

        The main training loop that iterates over epochs.

        Inside the loop: a train_fn to process one epoch of training data and a check_accuracy function to run validation and calculate metrics (using functions from metrics.py).

        Code to save the model checkpoints (e.g., saving the model with the best validation Dice score).

## evaluate.py

After training, you use this script to see how well your model performs on the unseen test set.

    Purpose: Test the final model performance.

    What it will contain:

        Code to load your best saved model from the saved_models/ directory.

        A DataLoader for the test dataset.

        A loop that iterates through the test data, generates predictions, and calculates the final scores (Dice, IoU, etc.) for the entire test set.

        It will print out a final report of the model's performance.

## predict.py

This script is for inference—using your trained model to segment a brand new, single MRI scan.

    Purpose: Generate a segmentation mask for a new, unseen scan.

    What it will contain:

        A function that takes the file paths to new T1, T2, and Flair scans as input.

        It will load the trained model.

        It will preprocess the input scans exactly like the training data.

        It passes the data through the model to get a raw prediction (a probability map).

        It post-processes the output (e.g., applies a threshold of 0.5) to create a binary mask.

        Finally, it saves the resulting segmentation mask as a new .nii file.

## utils.py

A place for all your helper functions to keep other scripts clean.

    Purpose: General utility functions.

    What it will contain:

        save_checkpoint(state, filename)

        load_checkpoint(checkpoint, model)

        Functions to plot and save examples of predictions vs. ground truth.

        Any other helper function you use in multiple places.