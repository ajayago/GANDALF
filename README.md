# GANDALF

This code repo has the code associated with the paper `GANDALF: Generative AttentioN based Data Augmentation and predictive modeLing Framework for personalized cancer treatment`.

GANDALF is a cancer drug response prediction model that addresses the problem of limited labelled patient data through a novel genomic data augmentation technique.

### Folders

* `Data` contains the notebooks needed to process the raw data files.
* `Processing` has files for additional processing such as annotations for mutations.
* `Model` has the relevant files for training GANDALF, along with relevant files for comparison with SOTA, ablation, sensitivity etc.
* `Evaluation` has notebooks for generating plots and visualizing results.

### Running GANDALF

Navigate to `Model` folder and perform the following steps.
1. Ensure that the correct values are filled up in `experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml`, especially `sample_id`, `input_data_type`, `feature_num` and `wandb_config` details.
2. Run step 1 and step 2 of GANDALF using 
```
python model_pretraining.py experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml
```
3. Use the saved augmented data to run steps 3, 4, 5 by running the notebook `mtl_pseudolabeling.ipynb` or `mtl_pseudolabeling-drug_specific.ipynb` as needed.