This folder houses the raw and intermediate data files for different datasets.

### Order to navigate files:
1. `1_raw_data_files.ipynb` loads all raw cell line and patient data and splits it into labelled and unlabelled datasets.
2. `2_train_test_splits.ipynb` creates 3 stratified folds over all datasets.

Other file available `deepchem_drug_feature_matrices.ipynb` used to preprocess drug data as needed by PANCDR baseline model.

Download the data files from `https://zenodo.org/records/13859069?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlmZWQyNTZkLTAwOGYtNDBiMS04YTU3LWY5MDQzMGRkZmViYiIsImRhdGEiOnt9LCJyYW5kb20iOiJjNDhjZWRkY2JhMmFmNzMwY2ZmYTk5NTNiMjdiMGVlZSJ9.oHUeorvGJVv3NqJ4IFkSAwzDtaPTlNxcFaJ4mRsT_ZO-6Uec1XVfML_IWblk-cTYEW4nasy_9Xeh_BQq38nmQA`. For training GANDALF, use pretrained transformer encoder from `pretrained_transformer_PREDICT_AI` folder and the train-test folds from `transformer_inputs_transformed_797`.