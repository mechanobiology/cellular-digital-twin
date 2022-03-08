This folder contains the code for the ML framework.
- aae.py: builds the ML framework and contains functions to train and test it
- data_reader.py: reads the data for training/testing the ML framework
- main.py: contains modifiable parameters for training/testing the ML framework
- ops.py: contains the architecture of the ML framework

This code was adapted from the cGAN developed in the paper "Computational Modeling of Cellular Structures Using Conditional Deep Generative Networks". The original code is available at: https://github.com/divelab/cgan

To train the ML framework the following command can be used: python main.py --action=train
To test the ML framework the following command can be used: python main.py --action=independent_test

The checkpoints from the training performed in the paper for FA and nucleus prediction can be downloaded from: https://drive.google.com/drive/folders/1I7DR0MXLonNME2ppwBrBVXOndn-wJBcW?usp=sharing

These checkpoints can be used to replicate the predictions obtained in the paper.
