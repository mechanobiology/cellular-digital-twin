This folder contains sample data to train/test ML framework.
- 3D_cells.zip: contains the 10 cells from independent test set. Contains ground truth and predicted cell morphology for building digital twin and true cell for comparison.
- train_set.zip: contains the 50 cells used for original training of ML framework divided into FA and nucleus subsets. These cells can be augmented and converted to HDF5 to replicate the training performed on the paper.
- test_set.zip: contains the 10 cells used for indipendent test of ML framework accuracy divided into FA and nucleus subsets. These cells can be converted to HDF5 to perform a prediction with trained ML framework. The results should be the same as seen in 3D_cells.zip predictions. 
