# Cellular Digital Twin
This repository contains the code and data for the paper "A Machine Learning Framework to Predict Subcellular Morphology of Endothelial Cells for Digital Twin Generation" currently under review. 

# Generating a Digital Twin
1. Download the training data from the train_set.zip contained in the data folder, which contains 50 single-cell stacks
2. Use the data_augmentation notebook to augment the dataset
3. Use the data_conversion notebook to convert the stacks to a HDF5 file
4. Train the ML framework (contained in 4 scripts in the ML_framework folder) using the command `python main.py --action=train`
5. Download the testing data from the test_set.zip contained in the data folder, which contains 10 single-cell stacks
6. Use the data_conversion notebook to convert the stacks to a HDF5 file
7. Run an independent test of the trained ML framework using the command `python main.py --action=independent_test`
8. Repeat steps 1-7 for both FA and Nucleus datasets
9. Compile FA and Nucleus results in a single folder (structured as shown in 3D_cells.zip in data folder)
10. Run 3D_cell_eval notebook to compute accuracy metrics and generate tiff files of true cell and digital twin
11. Open the Digital_Twin_True_Cell_Visualize.json file in AGAVE to visualize in 3D the true cell and digital twin (modify directory path in json file to visualize different cells)

# References
The ML framework code was adapted from the cGAN developed in the paper "Computational Modeling of Cellular Structures Using Conditional Deep Generative Networks": https://academic.oup.com/bioinformatics/article/35/12/2141/5162747

Please refer to their github repository for the original code: https://github.com/divelab/cgan

To visualize the cells in 3D, the AGAVE 3D pathtrace image viewer developed by the Allen Institute for Cell Science was used. To download AGAVE please go to: https://github.com/allen-cell-animated/agave/releases/tag/v1.2.4
