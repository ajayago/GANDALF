# Training of downstream DRP model with augmented data - first these new samples are pseudolabelled and then used for training.
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml
import pprint
import os
import wandb
import sys
import random
from scipy.stats import mode, pearsonr
import pickle
import itertools
# sys.path.append("./src/")

from src.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from src.modules import MLPDiffusion
from src.vae_model import vae
from src.loss_functions import get_kld_loss, coral
import sys
from model_definition import *

# global variables
assert len(sys.argv) >= 2, print("Please pass in the path to the model_config yaml!") # no args passed in

CONFIG_PATH = sys.argv[1] # model config path
pretty_print = pprint.PrettyPrinter()
print(f"Loading config from {CONFIG_PATH}")
config = yaml.safe_load(open(CONFIG_PATH))
model_config = config["model_hyperparams"]
folder_config = config["folder_config"]
wandb_config = config["wandb_config"]
wandb_config["project_name"] = wandb_config["project_name"] + f"-{model_config['experiment_id']}" # updates wandb project name for ease of monitoring and logging.
device = torch.device(f"cuda:{model_config['device']}" if torch.cuda.is_available() else "cpu")
genes_324 = list(pd.read_csv(f"{folder_config['data_folder']}/raw/metadata/gene2ind.txt", header=None)[0])
drug_fp = pd.read_csv(f"{folder_config['data_folder']}/raw/metadata/drug_morgan_fingerprints.csv", index_col=0)
suffixes = ["_piu_max", "_piu_sum", "_piu_mean", "_piu_count",
            "_lu_max", "_lu_sum", "_lu_mean", "_lu_count",
            "_ncu_max", "_ncu_sum", "_ncu_mean", '_ncu_count',
            "_pathogenic_max", "_pathogenic_sum", "_pathogenic_mean", "_pathogenic_count",
            "_vus_max", "_vus_sum", "_vus_mean", "_vus_count",
            "_benign_max", "_benign_sum", "_benign_mean", "_benign_count"
           ]
genes_7776 = []
for s in suffixes:
    for g in list(pd.read_csv(f"{folder_config['data_folder']}/raw/metadata/gene2ind.txt", header=None)[0]):
        genes_7776.append(f"{g}{s}")

# seeding
torch.manual_seed(model_config["seed"])
random.seed(model_config["seed"])
np.random.seed(model_config["seed"])
# reproducibility in data loading - https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(model_config["seed"])

# pass samples through the VAE and DDPM network, till just before VAE decoder
def vae_decoder_input(df, vae, diff_model):
    """
    Takes input df, pretrained vae and diffusion model as inputs, runs forward pass till VAE decoder
    """
    with torch.no_grad():
        batch = torch.tensor(df.values) # convert to torch tensor
        inp_vae = batch.to(device, dtype=torch.float32)
        inp, mu, logvar, _ = vae(inp_vae) # From VAE encoder + reparameterization
        
        noise = torch.randn_like(inp) # this is the label we use   
        b = inp.shape[0]
        t = (torch.ones((b,)) * 700).long().to(device) # fixing time steps to 700
        pt = torch.ones_like(t).float() / diff_model.num_timesteps
        inp_t = diff_model.gaussian_q_sample(inp, t, noise) # forward process with cell line model encoder
        
        model_out = diff_model._denoise_fn(inp_t, t) # predicted epsilon from patient decoder
    
        # predict inp from noise using patient model
        inp_pred = diff_model._predict_xstart_from_eps(inp_t, t, model_out)

    return inp_pred.detach().cpu().numpy()

def load_pretrained_models():
    pretrained = torch.load(f"{folder_config['model_checkpoint_folder']}/best_pretrained_validation_loss_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
    is_real = True if model_config["input_data_type"] == "binary_mutations" else False
    # patients
    patient_vae = vae(input_dim=model_config["feature_num"], k_list=model_config["patient_vae_k_list"], actf_list=model_config["patient_vae_actf_list"], is_real=is_real).to(device)
    tcga_mlp_diffusion_model = MLPDiffusion(d_in=model_config["patient_vae_k_list"][-1]//2, num_classes=0, is_y_cond=False, rtdl_params={"d_layers": [model_config["patient_vae_k_list"][-1]//4], "dropout": model_config["dropout"]}).to(device)
    tcga_diff_model = GaussianMultinomialDiffusion(num_classes=np.array([0]), num_numerical_features=model_config["patient_vae_k_list"][-1]//2, denoise_fn=tcga_mlp_diffusion_model, device=device)#.to(device)
    tcga_diff_model.load_state_dict(pretrained["patient_diff_model"])
    patient_vae.load_state_dict(pretrained["patient_vae_conditioned"])
    # cell lines
    cl_vae = vae(input_dim=model_config["feature_num"], k_list=model_config["cl_vae_k_list"], actf_list=model_config["cl_vae_actf_list"], is_real=is_real).to(device)
    cl_mlp_diffusion_model = MLPDiffusion(d_in=model_config["cl_vae_k_list"][-1]//2, num_classes=0, is_y_cond=False, rtdl_params={"d_layers": [model_config["cl_vae_k_list"][-1]//4], "dropout": model_config["dropout"]}).to(device)
    cl_diff_model = GaussianMultinomialDiffusion(num_classes=np.array([0]), num_numerical_features=model_config["cl_vae_k_list"][-1]//2, denoise_fn=cl_mlp_diffusion_model, device=device)#.to(device)
    cl_diff_model.load_state_dict(pretrained["cl_diff_model"])
    cl_vae.load_state_dict(pretrained["cl_vae_conditioned"])
    return cl_diff_model, cl_vae, tcga_diff_model, patient_vae

def load_datasets(sample_id):
    """
    Takes sample_id as input, loads source and target train, validation and test splits (predefined files from Processing folder).
    """
    data_dir = folder_config["data_folder"] + "input_types/"
    # navigate based on input type
    if model_config["input_data_type"] == "binary_mutations":
        data_dir = data_dir + "raw_mutations/"
        features2select = genes_324  # inclusive of Morgan drug fingerprints of 2048 dim
    elif model_config["input_data_type"] == "annotated_mutations":
        data_dir = data_dir + "annotated_mutations/"
        features2select = genes_7776  # inclusive of Morgan drug fingerprints of 2048 dim
    elif model_config["input_data_type"] == "transformer_inputs": # processed by PREDICT-AI transformer embedder
        data_dir = data_dir + "transformer_inputs_transformed_797/"
        features2select = [f"transformer_embedded_{i}" for i in range(797)] # after transformer embedding
    else:
        print("Unsupported input type!")
        return
    
    # navigate based on experiment id
    if model_config["experiment_id"] == "1A":
        data_dir = data_dir + "Experiment1/SettingA/"
    elif model_config["experiment_id"] == "1B":
        data_dir = data_dir + "Experiment1/SettingB/"
    elif model_config["experiment_id"] == "2A":
        data_dir = data_dir + "Experiment2/SettingA/"
    elif model_config["experiment_id"] == "2B":
        data_dir = data_dir + "Experiment2/SettingB/"
    else:
        print("Unsupported experiment ID!")
        return
    
    # load the fold based on sample_id - Note: cell lines have only 1 fold (fold 0)
    with open(f"{data_dir}/cell_lines_fold0_processed.pkl", "rb") as f:
        source_data = pickle.load(f)

    with open(f"{data_dir}/patients_fold{sample_id}_processed.pkl", "rb") as f:
        target_data = pickle.load(f)

    # load pretrained TCGA VAE and diffusion models
    # pass data points through patient DDPM and get the input to VAE decoder for DRP
    cl_diff_model, cl_vae, tcga_diff_model, patient_vae = load_pretrained_models()
    
    # select data based on experiment settings 
    # Can be CISPLATIN, PACLITAXEL, FLUOROURACIL, SORAFENIB for 1A, CISPLATIN, TCGA-CESC; CISPLATIN, TCGA-HNSC; PACLITAXEL, TCGA-BRCA; FLUOROURACIL, TCGA-STAD for 1B
    # ALL for 2A, TCGA-BRCA, TCGA-CESC, TCGA-HNSC, TCGA-STAD for 2B
    if model_config["experiment_id"] in ["1A", "2B"]:
        setting = model_config["experiment_settings"]
        train_source_data, val_source_data, test_source_data = source_data["train"][setting], source_data["val"][setting], source_data["test"][setting]
        train_target_data, val_target_data, test_target_data = target_data["train"][setting], target_data["val"][setting], target_data["test"][setting]
    elif model_config["experiment_id"] == "1B":
        setting = (model_config["experiment_settings"].split(", ")[0], model_config["experiment_settings"].split(", ")[1], "TCGA")
        train_source_data, val_source_data, test_source_data = source_data["train"][setting], source_data["val"][setting], source_data["test"][setting]
        train_target_data, val_target_data, test_target_data = target_data["train"][setting], target_data["val"][setting], target_data["test"][setting]
    elif model_config["experiment_id"] == "2A":
        train_source_data, val_source_data, test_source_data = source_data["train"], source_data["val"], source_data["test"]
        train_target_data, val_target_data, test_target_data = target_data["train"], target_data["val"], target_data["test"]
    else:
        print("Unsupported experiment settings and ID")
        return
    
    # merge dataframes with drug Morgan fingprint dataframes
    train_source_data_merged = train_source_data.merge(drug_fp, left_on="drug_name", right_on=drug_fp.index)
    val_source_data_merged = val_source_data.merge(drug_fp, left_on="drug_name", right_on=drug_fp.index)
    test_source_data_merged = test_source_data.merge(drug_fp, left_on="drug_name", right_on=drug_fp.index)

    train_target_data_merged = train_target_data.merge(drug_fp, left_on="drug_name", right_on=drug_fp.index)
    val_target_data_merged = val_target_data.merge(drug_fp, left_on="drug_name", right_on=drug_fp.index)
    test_target_data_merged = test_target_data.merge(drug_fp, left_on="drug_name", right_on=drug_fp.index)

    assert train_source_data_merged.shape[0] == train_source_data.shape[0], "Train source data loss after merge!"
    assert val_source_data_merged.shape[0] == val_source_data.shape[0], "Val source data loss after merge!"
    assert test_source_data_merged.shape[0] == test_source_data.shape[0], "Test source data loss after merge!"
    assert train_target_data_merged.shape[0] == train_target_data.shape[0], "Train target data loss after merge!"
    assert val_target_data_merged.shape[0] == val_target_data.shape[0], "Val target data loss after merge!"
    assert test_target_data_merged.shape[0] == test_target_data.shape[0], "Test target data loss after merge!"

    # separate out into input, drug and labels
    train_source_inputs, val_source_inputs, test_source_inputs = train_source_data_merged[features2select], val_source_data_merged[features2select], test_source_data_merged[features2select]
    # pass cl samples through cl diff model and vae
    train_source_inputs_vae = pd.DataFrame(vae_decoder_input(train_source_inputs, cl_vae, cl_diff_model), columns=[f"vae_feat{i}" for i in range(model_config["cl_vae_k_list"][-1]//2)], index=train_source_data_merged.index)
    val_source_inputs_vae = pd.DataFrame(vae_decoder_input(val_source_inputs, cl_vae, cl_diff_model), columns=[f"vae_feat{i}" for i in range(model_config["cl_vae_k_list"][-1]//2)], index=val_source_data_merged.index)
    test_source_inputs_vae = pd.DataFrame(vae_decoder_input(test_source_inputs, cl_vae, cl_diff_model), columns=[f"vae_feat{i}" for i in range(model_config["cl_vae_k_list"][-1]//2)], index=test_source_data_merged.index)
    train_source_drugs, val_source_drugs, test_source_drugs = train_source_data_merged[[str(i) for i in range(0, 2048)]].values, val_source_data_merged[[str(i) for i in range(0, 2048)]].values, test_source_data_merged[[str(i) for i in range(0, 2048)]].values
    train_source_labels, val_source_labels, test_source_labels = train_source_data_merged["auc"].values, val_source_data_merged["auc"].values, test_source_data_merged["auc"].values

    train_target_inputs, val_target_inputs, test_target_inputs = train_target_data_merged[features2select], val_target_data_merged[features2select], test_target_data_merged[features2select]
    # pass patient samples through tcga diff model and vae
    train_target_inputs_vae = pd.DataFrame(vae_decoder_input(train_target_inputs, patient_vae, tcga_diff_model), columns=[f"vae_feat{i}" for i in range(model_config["patient_vae_k_list"][-1]//2)], index=train_target_data_merged.sample_id)
    val_target_inputs_vae = pd.DataFrame(vae_decoder_input(val_target_inputs, patient_vae, tcga_diff_model), columns=[f"vae_feat{i}" for i in range(model_config["patient_vae_k_list"][-1]//2)], index=val_target_data_merged.sample_id)
    test_target_inputs_vae = pd.DataFrame(vae_decoder_input(test_target_inputs, patient_vae, tcga_diff_model), columns=[f"vae_feat{i}" for i in range(model_config["patient_vae_k_list"][-1]//2)], index=test_target_data_merged.sample_id)

    train_target_drugs, val_target_drugs, test_target_drugs = train_target_data_merged[[str(i) for i in range(0, 2048)]].values, val_target_data_merged[[str(i) for i in range(0, 2048)]].values, test_target_data_merged[[str(i) for i in range(0, 2048)]].values
    train_target_labels, val_target_labels, test_target_labels = train_target_data_merged["recist"].values, val_target_data_merged["recist"].values, test_target_data_merged["recist"].values

    return train_source_inputs_vae, train_source_drugs, train_source_labels, val_source_inputs_vae, val_source_drugs, val_source_labels, test_source_inputs_vae, test_source_drugs, test_source_labels, train_target_inputs_vae, train_target_drugs, train_target_labels, val_target_inputs_vae, val_target_drugs, val_target_labels, test_target_inputs_vae, test_target_drugs, test_target_labels, train_source_data_merged, val_source_data_merged, test_source_data_merged, train_target_data_merged, val_target_data_merged, test_target_data_merged

    # pass # needs to return (train_source_data, train_source_labels, val_source_data, val_source_labels, test_source_data, test_source_labels), (train_target_data, train_target_labels, val_target_data, val_target_labels, test_target_data, test_target_labels)
    #  Dummy data
    # train_source_data, val_source_data, test_source_data = np.random.rand(32, 2048 + 4), np.random.rand(10, 2048 + 4), np.random.rand(5, 2048 + 4)
    # train_source_labels, val_source_labels, test_source_labels = np.random.randint(2, size=32), np.random.randint(2, size=10), np.random.randint(2, size=5)
    # train_target_data, val_target_data, test_target_data = np.random.rand(32, 2048 + 4), np.random.rand(10, 2048 + 4), np.random.rand(3, 2048 + 4)
    # train_target_labels, val_target_labels, test_target_labels = np.random.randint(2, size=32), np.random.randint(2, size=10), np.random.randint(2, size=3)
    # return train_source_data, train_source_labels, val_source_data, val_source_labels, test_source_data, test_source_labels, train_target_data, train_target_labels, val_target_data, val_target_labels, test_target_data, test_target_labels

def load_augmented_cl_dataset(sample_id):
    augmented_cl_df = pd.read_csv(f"{folder_config['model_checkpoint_folder']}/augmented_cl_clconditioned_uda_v2_vaeinput_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.csv", index_col=0)
    print(f"Loaded augmented CL data: {augmented_cl_df.shape}")
    augmented_cl_df_test = pd.read_csv(f"{folder_config['model_checkpoint_folder']}/augmented_cl_test_clconditioned_uda_v2_vaeinput_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.csv", index_col=0)
    print(f"Loaded augmented CL Test data: {augmented_cl_df_test.shape}")
    return augmented_cl_df, augmented_cl_df_test

def inference_baseline_model(model, test_dataloader, domain="cell_line"):
    model.eval()
    y_test_pred = []
    test_y = []
    for idx, batch in enumerate(test_dataloader):
        y_test_pred.append(nn.Sigmoid()(model(batch[0].to(device, dtype=torch.float32))).detach())
        if len(batch) > 1: # has labels as well (i.e. patients not augmented cl)
            test_y.append(batch[1].detach())
    y_test_pred = torch.cat(y_test_pred)
    test_y = torch.cat(test_y)

    print(f"Saving predictions and gorund truth values to {folder_config['model_inference_folder']}/prediction_{domain}s_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.csv")
    res_df = pd.DataFrame()
    res_df["y_true"] = list(test_y.cpu().detach().numpy().reshape(-1))
    res_df["y_pred"] = list(y_test_pred.cpu().detach().numpy().reshape(-1))
    res_df.to_csv(f"{folder_config['model_inference_folder']}/prediction_{domain}s_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.csv")

# using data loaders to prevent execessive memory usage
class CustomCombinedDataSetLabelled(TensorDataset):
    def __init__(self, combined_df, cl_df, tcga_target_inputs_vae, drug_fp): # possible_combinations must only consist of samples with drug name with a fingerprint
        self.sample_df = combined_df.reset_index(drop=True)
        self.augmented_cl_df = cl_df
        self.tcga_vae_df = tcga_target_inputs_vae[~tcga_target_inputs_vae.index.duplicated(keep="first")]
        self.drug_fp = drug_fp

    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        sample_name = row["sample_id"]
        drug_name = row["drug_name"]
        if sample_name in self.tcga_vae_df.index: # using VAE version instead of mutation profiles
            mut_profile = self.tcga_vae_df.loc[sample_name].values
        if sample_name in self.augmented_cl_df.index:
            mut_profile = self.augmented_cl_df.loc[sample_name].values
        drug_inp = self.drug_fp.loc[drug_name].values
        inp = []
        inp.extend(mut_profile)
        inp.extend(drug_inp)
        if "recist" in row.index:
            response = row["recist"]
        elif "auc" in row.index:
            response = row["auc"]
        return torch.tensor(inp), response

    def __len__(self):
        return len(self.sample_df)
    
if __name__ == "__main__":
    print("-- Parameters used: --")
    pretty_print.pprint(model_config)

    # create folders to save trained model and predictions
    if not os.path.exists(folder_config['model_checkpoint_folder']):
        os.makedirs(folder_config['model_checkpoint_folder'])
        print(f"Created {folder_config['model_checkpoint_folder']}.")
    else:
        print(f"Directory {folder_config['model_checkpoint_folder']} already exists.")

    # load datasets
    train_source_inputs_vae, train_source_drugs, train_source_labels, val_source_inputs_vae, val_source_drugs, val_source_labels, test_source_inputs_vae, test_source_drugs, test_source_labels, train_target_inputs_vae, train_target_drugs, train_target_labels, val_target_inputs_vae, val_target_drugs, val_target_labels, test_target_inputs_vae, test_target_drugs, test_target_labels, train_source_data_merged, val_source_data_merged, test_source_data_merged, train_target_data_merged, val_target_data_merged, test_target_data_merged = load_datasets(model_config["sample_id"])

    # load augmented data for CL
    cl_augmented_df, cl_augmented_df_test = load_augmented_cl_dataset(model_config["sample_id"])

    # create data loader
    cl_test_dataset_df = test_source_data_merged
    cl_test_dataset = CustomCombinedDataSetLabelled(cl_test_dataset_df, cl_augmented_df_test, test_target_inputs_vae, drug_fp)
    cl_test_dataloader = DataLoader(cl_test_dataset, batch_size=model_config["drp_batch_size"], shuffle=False)

    patient_test_dataset_df = test_target_data_merged
    patient_test_dataset = CustomCombinedDataSetLabelled(patient_test_dataset_df, cl_augmented_df_test, test_target_inputs_vae, drug_fp)
    patient_test_dataloader = DataLoader(patient_test_dataset, batch_size=model_config["drp_batch_size"], shuffle=False)

    # initialise the DRP NN 
    nn_drp = BaseLineNN(2048 + model_config["patient_vae_k_list"][-1]//2, model_config["drp_hidden_dim"]).to(device)
    nn_drp.model_name = "DRP_model"
    # load pretrained
    nn_drp.load_state_dict(torch.load(f"{folder_config['model_checkpoint_folder']}/{nn_drp.model_name}_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth"))

    # run inference
    # cl
    inference_baseline_model(nn_drp, cl_test_dataloader, domain="cell_line")
    # patient
    inference_baseline_model(nn_drp, patient_test_dataloader, domain="patient")


