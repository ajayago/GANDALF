This folder houses the raw and intermediate data files for different datasets.

### Order to navigate files:
1. `1_raw_data_files.ipynb` loads all raw cell line and patient data and splits it into labelled and unlabelled datasets.
2. `2_train_test_splits.ipynb` creates 3 stratified folds over all datasets.

Other file available `deepchem_drug_feature_matrices.ipynb` used to preprocess drug data as needed by PANCDR baseline model.