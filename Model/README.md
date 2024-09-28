This folder has the code for all 5 steps of GANDALF, as organised below:

`model_pretraining.py` has code for 
1. Pretraining diffusion models
2. Generating augmented patient data from cell lines

`mtl_pseudolabeling.ipynb` has code for
3. Multi-task learning with labelled cell lines and patients
4. Pseudolabel assignment for generated patients from step 2, and selection of confident samples
5. Downsteam Drug Response Prediction Classifier training

`mtl_pseudolabeling-drug_specific.ipynb` helps fine-tune for a specific drug.

`model_definition.py` has the broad class definitions and `src` has relevant helper files and functions.

### Running GANDALF

1. Ensure that the correct values are filled up in `experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml`, especially `sample_id`, `input_data_type`, `feature_num` and `wandb_config` details.
2. Run step 1 and step 2 of GANDALF using 
```
python model_pretraining.py experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml
```
3. Use the saved augmented data to run steps 3, 4, 5 by running the notebook `mtl_pseudolabeling.ipynb` or `mtl_pseudolabeling-drug_specific.ipynb` as needed.

#### Ablation
Use the config files available in `experiment_settings_yaml/ablation/` and code from `ablation` for this.

|Setting | Pretraining | Training | Config file|
|--------|-------------|----------|-------------|
|`W/O MTL` | Same as GANDALF | `no_mtl.ipynb` | Same as GANDALF|
|`W/O cross-attention` | `model_pretraining_noattn.py` | `no_mtl-no_attn.ipynb` | `model_config_2A_annotated_mutations_v7_foldX_noattn.yaml` |
|`W/O transformer` | `model_pretraining_noattn-notransformer.py` | `no_mtl-no_attn-notransformer.ipynb` | `model_config_2A_annotated_mutations_v7_foldX_noattn_notransformer.yaml` |

#### Augmentation levels
Use config files `experiment_settings_yaml/ablation/model_config_2A_annotated_mutations_v7_foldX_noattn_notransformer.yaml` for this.

`baseline_gaussian_noise.ipynb` runs both vanilla DRP without any data augmentation (`W/O aug`) and with Gaussian perturbation (`W Gaussian`).

#### Pseudolabeling 
This compares GANDALF with a trinet like pseudolabeling approach. 

1. Ensure that the correct values are filled up in `experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml`, especially `sample_id`, `input_data_type`, `feature_num` and `wandb_config` details.
2. Run step 1 and step 2 of GANDALF using 
```
python model_pretraining.py experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml
```
3. Use the saved augmented data to run steps 3, 4, 5 by running 
```
python model_training_trinet.py experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml

python model_inference.py experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml

cd ../Evaluation/

python evaluation.py experiment_settings_yaml/model_config_2A_annotated_mutations_v7_foldX.yaml
```
