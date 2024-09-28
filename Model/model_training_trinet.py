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
import sys

from src.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from src.modules import MLPDiffusion
from src.vae_model import vae
from src.loss_functions import get_kld_loss, coral
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
wandb_config["project_name"] = wandb_config["project_name"] + f"-{model_config['experiment_id']}-{model_config['experiment_settings']}-fold{model_config['sample_id']}" # updates wandb project name for ease of monitoring and logging.
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

# setting up wandb
os.environ["WANDB_CACHE_DIR"] = wandb_config["wandb_cache_dir"]
os.environ["WANDB_DIR"] = wandb_config["wandb_cache_dir"]
wandb.login(key=wandb_config["api_key"])

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

    return train_source_inputs_vae, train_source_drugs, train_source_labels, val_source_inputs_vae, val_source_drugs, val_source_labels, test_source_inputs_vae, test_source_drugs, test_source_labels, train_target_inputs_vae, train_target_drugs, train_target_labels, val_target_inputs_vae, val_target_drugs, val_target_labels, test_target_inputs_vae, test_target_drugs, test_target_labels, train_target_data_merged, val_target_data_merged, test_target_data_merged

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
    return augmented_cl_df

def train_baseline_model(model, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    To train vanilla baseline model
    """
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    # training 
    val_losses = []
    count = 0
    for i in range(num_epochs):
        train_losses = []
        for idx, batch in enumerate(train_dataloader):
            optim.zero_grad()
            y_pred = model(batch[0].to(device, dtype=torch.float32))
            loss = criterion(y_pred, batch[1].view(-1, 1).to(device, dtype=torch.float32))
            train_losses.append(loss.detach().item())
            loss.backward()
            optim.step()
    
        with torch.no_grad():
            val_loss = 0.0
            for idx, batch in enumerate(val_dataloader):
                val_loss += criterion(model(batch[0].to(device, dtype=torch.float32)), batch[1].view(-1, 1).to(device, dtype=torch.float32))
            val_loss /= (idx+1)
    
        print(f"Epoch {i}: Training loss: {np.mean(train_losses)} |  Validation loss: {val_loss.detach().item()}")
        
        val_losses.append(val_loss.detach().item())

        wandb.log({
            f"{model.model_name}_train_loss": loss.detach().item(),
            f"{model.model_name}_val_loss": val_loss.detach().item()
        })
        if len(val_losses) == 1: # first epoch
            best_val_loss = val_loss
        
        if val_loss <= best_val_loss:
            best_val_score = val_loss
            # save model
            print("Best model")
            torch.save(model.state_dict(), f"{folder_config['model_checkpoint_folder']}/{model.model_name}_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
            count = 0 # reset count
        else:
            count += 1 # declining performance on validation data
        if count >= 3:
            print("Converged")
            break
        # # convergence checking based on validation loss
        # if len(val_losses) > 2:
        #     if val_losses[-1] > val_losses[-2]:
        #         count += 1
        #     else:
        #         print("Best model")
        #         torch.save(model.state_dict(), f"{folder_config['model_checkpoint_folder']}/{model.model_name}_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
        #         count = 0
        # if len(val_losses) == 1:
        #     torch.save(model.state_dict(), f"{folder_config['model_checkpoint_folder']}/{model.model_name}_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
        # if count > 3:
        #     print("Converged")
        #     break
    
    

def inference_baseline_model(model, test_dataloader):
    model.eval()
    y_test_pred = []
    test_y = []
    for idx, batch in enumerate(test_dataloader):
        # print(f"Batch id inference {idx}")
        y_test_pred.append(nn.Sigmoid()(model(batch.to(device, dtype=torch.float32))).detach())
        # print(y_test_pred[-1].shape)
    y_test_pred = torch.cat(y_test_pred)
    # test_y = torch.cat(test_y)
    print(y_test_pred.shape)
    return y_test_pred

def inference_drp_model(model, test_dataloader):
    model.eval()
    y_test_pred = []
    test_y = []
    for idx, batch in enumerate(test_dataloader):
        # print(f"Batch id inference {idx}")
        inp, label = batch
        y_test_pred.append(nn.Sigmoid()(model(inp.to(device, dtype=torch.float32))).detach())
        print(y_test_pred[-1].shape)
        # has labels as well (i.e. patients not augmented cl)
        test_y.append(label)
    y_test_pred = torch.cat(y_test_pred)
    test_y = torch.cat(test_y)
    print(y_test_pred.shape)
    return y_test_pred, test_y

def train_drp_model(model, train_dataloader, patient_val_dataloader, num_epochs=100, lr=1e-3):
    """
    To train vanilla baseline model
    """
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    # training 
    train_losses = []
    val_corrs = []
    count = 0
    for i in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            optim.zero_grad()
            y_pred = model(batch[0].to(device, dtype=torch.float32))
            loss = criterion(y_pred, batch[1].view(-1, 1).to(device, dtype=torch.float32))
            loss.backward()
            optim.step()

        y_test_pred, test_y = inference_drp_model(model, patient_val_dataloader)
        patient_corr = pearsonr(test_y.detach().cpu().numpy().reshape(-1), y_test_pred.detach().cpu().numpy().reshape(-1)).statistic + 1 # range in [0, 2]

        val_corrs.append(patient_corr)
        print(f"Epoch {i}: Training loss: {loss.detach().item()} |  Validation correlation: {patient_corr}")

        wandb.log({
            f"{model.model_name}_train_loss": loss.detach().item(),
            f"validation_score": patient_corr
        })
        # convergence based on val score
        if len(val_corrs) == 1: # first epoch
            best_val_score = patient_corr

        # save model
        if model_config["model_save_criteria"] in ["val_AUROC", "val_AUPRC", "val_corr"]: # maximise values
            if patient_corr >= best_val_score:
                best_val_score = patient_corr
                # save model
                print("Best model")
                torch.save(model.state_dict(), f"{folder_config['model_checkpoint_folder']}/{model.model_name}_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
                count = 0 # reset count
            else:
                count += 1 # declining performance on validation data
        else:
            print("Unsupported metric for optimising")
            return
        
        if count >= 3:
            print("Converged")
            break

        # # convergence checking based on validation correlation
        # if len(val_corrs) > 2:
        #     if val_corrs[-1] < val_corrs[-2]: # maximise correlation
        #         count += 1
        #     else:
        #         print("Best model")
        #         torch.save(model.state_dict(), f"{folder_config['model_checkpoint_folder']}/{model.model_name}_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
        #         count = 0
        # if len(val_corrs) == 1:
        #     torch.save(model.state_dict(), f"{folder_config['model_checkpoint_folder']}/{model.model_name}_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
        # if count > 3:
        #     print("Converged")
        #     break
    


        
    
# using data loaders to prevent execessive memory usage
class CustomCellLineDataSetUnlabelled(TensorDataset):
    def __init__(self, cl_augmented_df, drug_fp, possible_combinations): # possible_combinations must only consist of samples with drug name with a fingerprint
        self.possible_combinations = possible_combinations
        self.augmented_cl_df = cl_augmented_df
        self.drug_fp = drug_fp

    def __getitem__(self, idx):
        sample_name, drug_name = self.possible_combinations[idx]
        mut_profile = self.augmented_cl_df.loc[sample_name].values
        drug_inp = self.drug_fp.loc[drug_name].values
        inp = []
        inp.extend(mut_profile)
        inp.extend(drug_inp)
        return torch.tensor(inp)

    def __len__(self):
        return len(self.possible_combinations)
    
# using data loaders to prevent execessive memory usage
class CustomCombinedDataSetLabelled(TensorDataset):
    def __init__(self, combined_df, cl_augmented_df, train_target_inputs_vae, drug_fp): # possible_combinations must only consist of samples with drug name with a fingerprint
        self.sample_df = combined_df.reset_index(drop=True)
        self.augmented_cl_df = cl_augmented_df
        self.tcga_vae_df = train_target_inputs_vae[~train_target_inputs_vae.index.duplicated(keep="first")]
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
        response = row["recist"]
        return torch.tensor(inp), response

    def __len__(self):
        return len(self.sample_df)
        
def convert_binary(prediction, lower_threshold, upper_threshold):
    if prediction >= upper_threshold:
        return 1
    elif prediction < lower_threshold:
        return 0
    else:
        return -1
    
def majority_vote(row):
    values = []
    for i in row.index:
        if "preds_binary" in i:
            values.append(row[i])
    mod = mode(values).mode
    return mod
    




if __name__ == "__main__":
    run_wb = wandb.init(
        project=wandb_config["project_name"],
        config=model_config,
    )
    print(f"Running WANDB run - {run_wb.name} for project {wandb_config['project_name']}")
    print("-- Parameters used: --")
    pretty_print.pprint(model_config)

    # create folders to save trained model and predictions
    if not os.path.exists(folder_config['model_checkpoint_folder']):
        os.makedirs(folder_config['model_checkpoint_folder'])
        print(f"Created {folder_config['model_checkpoint_folder']}.")
    else:
        print(f"Directory {folder_config['model_checkpoint_folder']} already exists.")

    # load datasets
    train_source_inputs_vae, train_source_drugs, train_source_labels, val_source_inputs_vae, val_source_drugs, val_source_labels, test_source_inputs_vae, test_source_drugs, test_source_labels, train_target_inputs_vae, train_target_drugs, train_target_labels, val_target_inputs_vae, val_target_drugs, val_target_labels, test_target_inputs_vae, test_target_drugs, test_target_labels, train_target_data_merged, val_target_data_merged, test_target_data_merged = load_datasets(model_config["sample_id"])

    # create datasets
    # Cell Lines
    train_source_data = np.concatenate((train_source_inputs_vae.values, train_source_drugs), axis=1)
    val_source_data = np.concatenate((val_source_inputs_vae.values, val_source_drugs), axis=1)
    test_source_data = np.concatenate((test_source_inputs_vae.values, test_source_drugs), axis=1)
    source_dataset_train = TensorDataset(torch.FloatTensor(train_source_data), torch.FloatTensor(train_source_labels))
    source_dataset_val = TensorDataset(torch.FloatTensor(val_source_data), torch.FloatTensor(val_source_labels))
    source_dataset_test = TensorDataset(torch.FloatTensor(test_source_data), torch.FloatTensor(test_source_labels))

    # Patients
    train_target_data = np.concatenate((train_target_inputs_vae.values, train_target_drugs), axis=1)
    val_target_data = np.concatenate((val_target_inputs_vae.values, val_target_drugs), axis=1)
    test_target_data = np.concatenate((test_target_inputs_vae.values, test_target_drugs), axis=1)
    target_dataset_train = TensorDataset(torch.FloatTensor(train_target_data), torch.FloatTensor(train_target_labels.astype(int)))
    target_dataset_val = TensorDataset(torch.FloatTensor(val_target_data), torch.FloatTensor(val_target_labels.astype(int)))
    target_dataset_test = TensorDataset(torch.FloatTensor(test_target_data), torch.FloatTensor(test_target_labels.astype(int)))

    # create dataloaders
    source_dataloader_train = DataLoader(source_dataset_train, batch_size = model_config["source_batch_size"], shuffle = True, worker_init_fn = seed_worker, generator = g)
    source_dataloader_val = DataLoader(source_dataset_val, batch_size = model_config["source_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)
    source_dataloader_test = DataLoader(source_dataset_test, batch_size = model_config["source_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)

    target_dataloader_train = DataLoader(target_dataset_train, batch_size = model_config["target_batch_size"], shuffle = True, worker_init_fn = seed_worker, generator = g)
    target_dataloader_val = DataLoader(target_dataset_val, batch_size = model_config["target_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)
    target_dataloader_test = DataLoader(target_dataset_test, batch_size = model_config["target_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)

    # instantiate 3 baseline NNs to be trained on just patient data
    # define k different NNs to get a majority vote later. Dims: 2048 for drug fp, rest for sample representation
    nn_baselines = {}
    for i, k in enumerate(model_config["baseline_model_hidden_dim_list"]):
        nn_baselines[i] = BaseLineNN(2048 + model_config["patient_vae_k_list"][-1]//2, k).to(device)
        nn_baselines[i].model_name = f"BaselineNN_{i}"

    # train baseline models with patient data
    for i in nn_baselines.keys():
        train_baseline_model(model=nn_baselines[i], train_dataloader=target_dataloader_train, val_dataloader=target_dataloader_val, num_epochs=model_config["baseline_patient_epochs"], lr=model_config["baseline_patient_lr"])
    
    # use each trained baseline model to generate predictions for augmented cell line data
    # first generate all possible cell line-drug combinations for which pseudo label is to be generated
    cl_augmented_df = load_augmented_cl_dataset(model_config["sample_id"])
    train_val_cell_lines = list(cl_augmented_df.index)
    if model_config["experiment_id"] == "1B":
        drugs_with_fp = [model_config["experiment_settings"].split(", ")[0]] # extract out drug name
    elif model_config["experiment_id"] == "1A":
        drugs_with_fp = [model_config["experiment_settings"]] # has only drug name
    else: # in 2A and 2B include all available drugs with fp
        drugs_with_fp = list(drug_fp.index)
    possible_cl_drug_combinations = list(itertools.product(train_val_cell_lines, drugs_with_fp))
    possible_cl_drug_combinations_df = pd.DataFrame(possible_cl_drug_combinations, columns = ["sample_id", "drug_name"])

    cl_aug_train_dataset = CustomCellLineDataSetUnlabelled(cl_augmented_df, drug_fp, possible_cl_drug_combinations)
    print("Number of possible cl drug combos before pseudo label based filtering: ")
    print(len(cl_aug_train_dataset))
    cl_aug_train_dataloader = DataLoader(cl_aug_train_dataset, batch_size=model_config["source_batch_size"], shuffle=False) # to preserve order for later subset selection

    # assign pseudo labels from each baseline nn
    pseudolabels_df = pd.DataFrame()
    for i in nn_baselines.keys():
        cl_pseudolabels = inference_baseline_model(nn_baselines[i], cl_aug_train_dataloader)
        pseudolabels_df = pd.concat([pseudolabels_df, pd.Series(cl_pseudolabels.detach().cpu().numpy().reshape(-1))], axis = 1)        
    pseudolabels_df.columns = [f"baseline{i}_preds" for i in range(pseudolabels_df.shape[1])]
    

    # convert pseudolabels to binary labels
    for c in pseudolabels_df.columns:
        pseudolabels_df[f"{c}_binary"] = pseudolabels_df[c].apply(lambda x: convert_binary(x, model_config["pseudolabel_lower_threshold"], model_config["pseudolabel_upper_threshold"]))

    # get majority vote label
    pseudolabels_df["majority_vote_pseudolabels"] = pseudolabels_df[[c for c in pseudolabels_df.columns if "preds_binary" in c]].apply(lambda x: majority_vote(x), axis =1)
    pseudolabels_df.to_csv(f"{folder_config['model_checkpoint_folder']}/pseudolabels_df_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.csv")
    # non-abstained, confident pseudo labels
    confident_pseudolabels_df = pseudolabels_df[pseudolabels_df.majority_vote_pseudolabels != -1]
    confident_pseudolabels_df_idx = confident_pseudolabels_df.index # used to filter out the possible drug combinations df

    confident_cl_drug_combinations_df = possible_cl_drug_combinations_df[possible_cl_drug_combinations_df.index.isin(confident_pseudolabels_df_idx)].copy()
    confident_cl_drug_combinations_df["recist"] = list(confident_pseudolabels_df["majority_vote_pseudolabels"])
    print("Number of confident cl drug combinations with pseudolabels: ")
    print(confident_cl_drug_combinations_df.shape)
    print("Pseudo label distribution after majority vote:")
    print(confident_cl_drug_combinations_df.recist.value_counts())

    # combine confident CL samples with pseudolabels, with TCGA train data
    combined_dataset_df = pd.concat([confident_cl_drug_combinations_df, train_target_data_merged[confident_cl_drug_combinations_df.columns]], axis=0)
    combined_dataset = CustomCombinedDataSetLabelled(combined_dataset_df, cl_augmented_df, train_target_inputs_vae, drug_fp)
    combined_dataloader = DataLoader(combined_dataset, batch_size=model_config["drp_batch_size"], shuffle=True)

    # initialise the DRP NN 
    nn_drp = BaseLineNN(2048 + model_config["patient_vae_k_list"][-1]//2, model_config["drp_hidden_dim"]).to(device)
    nn_drp.model_name = "DRP_model"

    # Train DRP model
    train_drp_model(nn_drp, combined_dataloader, target_dataloader_val, num_epochs=model_config["drp_epochs"], lr=model_config["drp_lr"])

    wandb.finish() # exit run and upload logs






    