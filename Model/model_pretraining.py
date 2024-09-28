# Pretraining of VAEs and LDM model
import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
import yaml
import pprint
import os
import wandb
import sys
import random
import pickle
# sys.path.append("./src/")
import sys

from src.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from src.modules import MLPDiffusion
from src.vae_model import vae
from src.loss_functions import get_kld_loss, coral

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

def load_unlabelled_datasets(sample_id):
    """
    Takes sample_id as input, loads source and target train, validation and test splits (predefined files from Processing folder).
    """
    data_dir = folder_config["data_folder"] + "input_types/"
    # navigate based on input type
    if model_config["input_data_type"] == "binary_mutations":
        data_dir = data_dir + "raw_mutations/"
        features2select = genes_324 + ["sample_id"]# does not include Morgan drug fingerprints of 2048 dim
    elif model_config["input_data_type"] == "annotated_mutations":
        data_dir = data_dir + "annotated_mutations/"
        features2select = genes_7776 + ["sample_id"]# does not include Morgan drug fingerprints of 2048 dim
    elif model_config["input_data_type"] == "transformer_inputs": # processed by PREDICT-AI transformer embedder
        data_dir = data_dir + "transformer_inputs_transformed_797/"
        features2select = [f"transformer_embedded_{i}" for i in range(797)] + ["sample_id"] # after transformer embedding
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

    # separate out into input and labels
    train_source_inputs, val_source_inputs, test_source_inputs = train_source_data[features2select].drop_duplicates(), val_source_data[features2select].drop_duplicates(), test_source_data[features2select].drop_duplicates()
    # train_source_labels, val_source_labels, test_source_labels = train_source_data_merged["auc"].values, val_source_data_merged["auc"].values, test_source_data_merged["auc"].values

    train_target_inputs, val_target_inputs, test_target_inputs = train_target_data[features2select].drop_duplicates(), val_target_data[features2select].drop_duplicates(), test_target_data[features2select].drop_duplicates()
    # train_target_labels, val_target_labels, test_target_labels = train_target_data_merged["recist"].values, val_target_data_merged["recist"].values, test_target_data_merged["recist"].values

    return train_source_inputs, val_source_inputs, test_source_inputs, train_target_inputs, val_target_inputs, test_target_inputs

    # pass # needs to return (train_source_data, train_source_labels, val_source_data, val_source_labels, test_source_data, test_source_labels), (train_target_data, train_target_labels, val_target_data, val_target_labels, test_target_data, test_target_labels)
    #  Dummy data
    # train_source_data, val_source_data, test_source_data = np.random.rand(32, 4), np.random.rand(10, 4), np.random.rand(5, 4)
    # train_source_labels, val_source_labels, test_source_labels = np.random.randint(2, size=32), np.random.randint(2, size=10), np.random.randint(2, size=5)
    # train_target_data, val_target_data, test_target_data = np.random.rand(32, 4), np.random.rand(10, 4), np.random.rand(3, 4)
    # train_target_labels, val_target_labels, test_target_labels = np.random.randint(2, size=32), np.random.randint(2, size=10), np.random.randint(2, size=3)
    # return train_source_data, train_source_labels, val_source_data, val_source_labels, test_source_data, test_source_labels, train_target_data, train_target_labels, val_target_data, val_target_labels, test_target_data, test_target_labels

### Pretraining Stage 1 functions: Pretrain VAEs used for obtaining latent space ###
def testing_loop_vae(test_dataloader, vae, device):
    """
    Evaluation loop for VAE training to reconstruct input data
    """
    avg_test_loss = 0.0
    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            inp_vae = batch[0].to(device=device, dtype=torch.float32)
            inp, mu, logvar, inp_dec  = vae(inp_vae) # From VAE encoder + reparameterization

            mse_recon_loss = torch.nn.MSELoss()(inp_dec, inp_vae) # MSE 
            avg_test_loss += mse_recon_loss.detach().item()

    return avg_test_loss/(idx+1)

def training_loop_vae(train_dataloader, val_dataloader, epochs, vae, device, optimizer, scheduler=None, domain="patient"):
    train_losses = []
    val_scores = []
    early_stopping_pretrain_count = 0
    for i in range(epochs):
        loss = 0.0
        avg_test_loss = ""
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            inp_vae = batch[0].to(device=device, dtype=torch.float32)
            inp, mu, logvar, inp_dec = vae(inp_vae) # From VAE encoder + reparameterization
            mse_recon_loss = torch.nn.MSELoss()(inp_dec, inp_vae) # MSE 
            kld_loss = get_kld_loss(mu, logvar, is_mean=True) # KLD
            total_loss = mse_recon_loss + kld_loss

            total_loss.backward()
            optimizer.step()

            loss += total_loss.detach().item()

        avg_test_loss = testing_loop_vae(val_dataloader, vae, device)

        print(f"|Epoch {i} | Average training loss: {loss/((idx+1))} |{avg_test_loss}")
        train_losses.append(loss/(idx+1))
        
        if scheduler:
            scheduler.step()
        
        val_scores.append(avg_test_loss)
        wandb.log({
            f"pretrain_loss_total_{domain}": loss/(idx+1),
            f"avg_val_loss_{domain}": avg_test_loss
        })

        if len(val_scores) == 1: # first epoch
            best_val_score = avg_test_loss

        # save model
        if avg_test_loss <= best_val_score: # only checks val loss
            best_val_score = avg_test_loss
            # save best model
            torch.save({
                "config": model_config,
                "vae": vae.state_dict(),
                "epoch_num": i,
                "best_val_score": best_val_score
            }, f"{folder_config['model_checkpoint_folder']}/best_pretrained_vae_validation_loss_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}_{domain}.pth")
            print("Saved Model!")
            early_stopping_pretrain_count = 0
        else:
            early_stopping_pretrain_count += 1

        if early_stopping_pretrain_count >= 3:
            break



### Pretraining Stage 2 functions: Joint training using LDM and both domains. Includes cell line based conditioning. ###
# testing function with LDM using patient val data, based on reconstruction
def testing_loop_ldm(test_dataloader, diffusion_model, vae, device):
    avg_test_loss = 0.0
    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            inp_vae = batch[0].to(device=device, dtype=torch.float32)
            inp, mu, logvar, _ = vae(inp_vae) # From VAE encoder + reparameterization
            
            noise = torch.randn_like(inp).to(device) # this is the label we use   
            b = inp.shape[0]
            t, pt = diffusion_model.sample_time(b, device, 'uniform')
            inp_t = diffusion_model.gaussian_q_sample(inp, t, noise) # forward process
            model_out = diffusion_model._denoise_fn(inp_t, t) # predicted epsilon

            # MSE between predicted inp_pred after reverse diffusion and original input
            inp_pred = diffusion_model._predict_xstart_from_eps(inp_t, t, model_out)
            out_vae = vae.decoder(inp_pred) # after VAE decoding
            # X_mu1, X_theta1, X_pi1 = ffn_zinb_model(out_vae) # ZINorm
            mse_recon_loss = torch.nn.MSELoss()(out_vae, inp_vae) # MSE 

            avg_test_loss += mse_recon_loss.detach().item()

    return avg_test_loss/(idx+1)

# training function with LDM with CORAL loss
def training_loop_ldm_coral_conditioning(train_dataloaders, val_dataloaders, epochs, diffusion_model, vae, device, optimizer, scheduler=None):
    train_losses = []
    val_scores = []
    early_stopping_pretrain_count = 0

    for i in range(epochs):
        loss = 0.0
        avg_test_loss = ""
        for idx0, batch0 in enumerate(train_dataloaders["cell_line"]):
            for idx1, batch1 in enumerate(train_dataloaders["patient"]):
                optimizer.zero_grad()
                # for cell lines
                inp_vae_cl = batch0[0].to(device=device, dtype=torch.float32)
                inp_cl, mu_cl, logvar_cl, _ = vae["cell_line"](inp_vae_cl) # From VAE encoder + reparameterization
                
                noise_cl = torch.randn_like(inp_cl).to(device) # this is the label we use   
                b = inp_cl.shape[0]
                t, pt = diffusion_model["cell_line"].sample_time(b, device, 'uniform')
                inp_t_cl = diffusion_model["cell_line"].gaussian_q_sample(inp_cl, t, noise_cl) # forward process
                model_out = diffusion_model["cell_line"]._denoise_fn(inp_t_cl, t) # predicted epsilon
                # loss calculation using MSE between predicted and actual noise
                loss_gauss_cl = diffusion_model["cell_line"]._gaussian_loss(model_out, inp_cl, inp_t_cl, t, noise_cl).mean()
    
                # reconstruct inp, pass through VAE decoder and calculate VAE losses
                inp_pred_cl = diffusion_model["cell_line"]._predict_xstart_from_eps(inp_t_cl, t, model_out)
                out_vae_cl = vae["cell_line"].decoder(inp_pred_cl) # after VAE decoding
                mse_recon_loss_cl = torch.nn.MSELoss()(out_vae_cl, inp_vae_cl)
                kld_loss_cl = get_kld_loss(mu_cl, logvar_cl, is_mean=True)

                # for patients
                inp_vae_tcga = batch1[0].to(device=device, dtype=torch.float32)
                inp_tcga, mu_tcga, logvar_tcga, _ = vae["patient"](inp_vae_tcga) # From VAE encoder + reparameterization
                
                noise_tcga = torch.randn_like(inp_tcga).to(device) # this is the label we use   
                b = inp_tcga.shape[0]
                t, pt = diffusion_model["patient"].sample_time(b, device, 'uniform')
                inp_t_tcga = diffusion_model["patient"].gaussian_q_sample(inp_tcga, t, noise_tcga) # forward process
                
                model_out = diffusion_model["patient"]._denoise_fn(inp_t_tcga, t) # predicted epsilon
                # loss calculation using MSE between predicted and actual noise
                loss_gauss_tcga = diffusion_model["patient"]._gaussian_loss(model_out, inp_tcga, inp_t_tcga, t, noise_tcga).mean()
    
                # reconstruct inp, pass through VAE decoder and calculate VAE losses
                inp_pred_tcga = diffusion_model["patient"]._predict_xstart_from_eps(inp_t_tcga, t, model_out)
                out_vae_tcga = vae["patient"].decoder(inp_pred_tcga) # after VAE decoding
                mse_recon_loss_tcga = torch.nn.MSELoss()(out_vae_tcga, inp_vae_tcga)
                kld_loss_tcga = get_kld_loss(mu_tcga, logvar_tcga, is_mean=True)
                
                # CORAL loss between latent spaces of VAEs - change to latnet space of diffusion models?
                # coral_loss = coral(inp_cl, inp_tcga)
                coral_loss = coral(inp_t_cl, inp_t_tcga)
                
                # cross attention between inp_t_cl from cell line forward process and inp_t_tcga from patient forward process
                # perform cross attention between cell lines and patients in the VAE latent space 
                # forward pass this attention vector through patient DDPM decoder
                # check KL div loss between the patient data original and those with cross attention
                query =  inp_t_tcga
                key = inp_t_cl
                value = inp_t_cl

                cross_attention_tcga = torch.nn.Softmax(dim=1)(query @ key.transpose(0, 1))/math.sqrt(query.shape[1]) @ value # same shape as query
                model_out_cross = diffusion_model["patient"]._denoise_fn(cross_attention_tcga, t) # predicted epsilon
                
                # loss calculation using KL div between predicted and actual noise
                loss_gauss_tcga_cross = diffusion_model["patient"]._vb_terms_bpd(
                                            model_output=model_out_cross,
                                            x_start=inp_tcga,
                                            x_t=inp_t_tcga,
                                            t=t,
                                            clip_denoised=False,
                                            model_kwargs=None,
                                        )["output"].mean()
                #diffusion_model["patient"]._gaussian_loss(model_out, inp_tcga, inp_t_tcga, t, noise_tcga).mean()
                
                total_loss = loss_gauss_cl + loss_gauss_tcga + coral_loss + mse_recon_loss_cl + mse_recon_loss_tcga + kld_loss_cl + kld_loss_tcga + loss_gauss_tcga_cross
                
                total_loss.backward()
                optimizer.step()

                # print(total_loss.detach().item())
                loss += total_loss.detach().item()

        avg_test_loss = testing_loop_ldm(val_dataloaders['patient'], diffusion_model['patient'], vae['patient'], device)

        print(f"|Epoch {i} | Average training loss: {loss/((idx0+1) * (idx1+1))} |{avg_test_loss}")
        train_losses.append(loss/((idx0+1) * (idx1+1)))
        
        if scheduler:
            scheduler.step()

        val_scores.append(avg_test_loss)
        wandb.log({
            "pretrain_loss_total_joint_training": loss/((idx0+1) * (idx1+1)),
            "avg_val_loss_joint_training": avg_test_loss
        })

        if len(val_scores) == 1: # first epoch
            best_val_score = avg_test_loss

        # save model
        if avg_test_loss <= best_val_score: # only checks val loss
            best_val_score = avg_test_loss
            # save best model
            torch.save({
                "config": model_config,
                "cl_diff_model": diffusion_model["cell_line"].state_dict(),
                "patient_diff_model": diffusion_model["patient"].state_dict(),
                "cl_vae_conditioned": vae["cell_line"].state_dict(),
                "patient_vae_conditioned": vae["patient"].state_dict(),
                "epoch_num": i,
                "best_val_score": best_val_score
            }, f"{folder_config['model_checkpoint_folder']}/best_pretrained_validation_loss_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.pth")
            print("Saved Model!")
            early_stopping_pretrain_count = 0
        else:
            early_stopping_pretrain_count += 1

        if early_stopping_pretrain_count >= 3:
            break

### Pretraining Stage 3: Augmenting existing dataset (unlabelled) using available cell line data ###
def augmented_cl(dataloader, cl_vae, cl_diff_model, tcga_diff_model):
    augmented = []
    for idx, batch in enumerate(dataloader):
        inp_vae = batch[0].to(device, dtype=torch.float32)
        inp, mu, logvar, _ = cl_vae(inp_vae) # From VAE encoder + reparameterization
        
        noise = torch.randn_like(inp) # this is the label we use   
        b = inp.shape[0]
        # t, pt = diffusion_model["cell_line"].sample_time(b, device, 'uniform')
        t = (torch.ones((b,)) * 700).long().to(device) # fixing time steps to 700
        pt = torch.ones_like(t).float() / cl_diff_model.num_timesteps
        inp_t = cl_diff_model.gaussian_q_sample(inp, t, noise) # forward process with cell line model encoder
        
        model_out = tcga_diff_model._denoise_fn(inp_t, t) # predicted epsilon from patient decoder
    
        # predict inp from noise using patient model
        inp_pred = tcga_diff_model._predict_xstart_from_eps(inp_t, t, model_out)
        # out_vae = patient_vae.decoder(inp_pred) # after VAE decoding with patient decoder
        # augmented.append(out_vae)
        augmented.append(inp_pred) # using VAE output instead.
    return torch.cat(augmented, axis = 0)
        
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

    # load data
    train_source_data, val_source_data, test_source_data, train_target_data, val_target_data, test_target_data = load_unlabelled_datasets(sample_id=model_config["sample_id"])

    # create datasets
    # Cell Lines
    source_dataset_train = TensorDataset(torch.FloatTensor(train_source_data.drop(["sample_id"], axis=1).values))
    source_dataset_val = TensorDataset(torch.FloatTensor(val_source_data.drop(["sample_id"], axis=1).values))
    source_dataset_test = TensorDataset(torch.FloatTensor(test_source_data.drop(["sample_id"], axis=1).values))

    # Patients
    target_dataset_train = TensorDataset(torch.FloatTensor(train_target_data.drop(["sample_id"], axis=1).values))
    target_dataset_val = TensorDataset(torch.FloatTensor(val_target_data.drop(["sample_id"], axis=1).values))
    target_dataset_test = TensorDataset(torch.FloatTensor(test_target_data.drop(["sample_id"], axis=1).values))

    # create dataloaders
    source_dataloader_train = DataLoader(source_dataset_train, batch_size = model_config["source_batch_size"], shuffle = True, worker_init_fn = seed_worker, generator = g)
    source_dataloader_val = DataLoader(source_dataset_val, batch_size = model_config["source_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)
    source_dataloader_test = DataLoader(source_dataset_test, batch_size = model_config["source_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)

    target_dataloader_train = DataLoader(target_dataset_train, batch_size = model_config["target_batch_size"], shuffle = True, worker_init_fn = seed_worker, generator = g)
    target_dataloader_val = DataLoader(target_dataset_val, batch_size = model_config["target_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)
    target_dataloader_test = DataLoader(target_dataset_test, batch_size = model_config["target_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)

    # Instantiate VAE models for each domain
    is_real = True if model_config["input_data_type"] == "binary_mutations" else False
    # cell line VAE
    cl_vae = vae(input_dim=model_config["feature_num"], k_list=model_config["cl_vae_k_list"], actf_list=model_config["cl_vae_actf_list"], is_real=is_real).to(device)
    optim_cl = torch.optim.Adam(cl_vae.parameters(), lr = model_config["cl_pretrain_lr"])

    # patient VAE
    patient_vae = vae(input_dim=model_config["feature_num"], k_list=model_config["patient_vae_k_list"], actf_list=model_config["patient_vae_actf_list"], is_real=is_real).to(device)
    optim_tcga = torch.optim.Adam(patient_vae.parameters(), lr = model_config["patient_pretrain_lr"])

    # Train VAEs to reconstruct each domain separately
    training_loop_vae(source_dataloader_train, source_dataloader_val, model_config["cl_pretrain_epochs"], cl_vae, device, optim_cl, domain="cell_line")
    training_loop_vae(target_dataloader_train, target_dataloader_val, model_config["patient_pretrain_epochs"], patient_vae, device, optim_tcga, domain="patient")

    # Joint training after freezing VAE parameters
    # freeze params of VAEs (pretrained)
    for param in cl_vae.parameters():
        param.requires_grad = False

    for param in patient_vae.parameters():
        param.requires_grad = False

    vaes = {
        "cell_line": cl_vae,
        "patient": patient_vae
    }

    # instantiate models
    # cell line
    cl_mlp_diffusion_model = MLPDiffusion(d_in=model_config["cl_vae_k_list"][-1]//2, num_classes=0, is_y_cond=False, rtdl_params={"d_layers": [model_config["cl_vae_k_list"][-1]//4], "dropout": model_config["dropout"]}).to(device)
    cl_diff_model = GaussianMultinomialDiffusion(num_classes=np.array([0]), num_numerical_features=model_config["cl_vae_k_list"][-1]//2, denoise_fn=cl_mlp_diffusion_model, device=device)#.to(device)
    # patient
    tcga_mlp_diffusion_model = MLPDiffusion(d_in=model_config["patient_vae_k_list"][-1]//2, num_classes=0, is_y_cond=False, rtdl_params={"d_layers": [model_config["patient_vae_k_list"][-1]//4], "dropout": model_config["dropout"]}).to(device)
    tcga_diff_model = GaussianMultinomialDiffusion(num_classes=np.array([0]), num_numerical_features=model_config["patient_vae_k_list"][-1]//2, denoise_fn=tcga_mlp_diffusion_model, device=device)#.to(device)

    diffusion_models = {
        "cell_line": cl_diff_model,
        "patient": tcga_diff_model
    }

    train_dataloaders = {
        "cell_line": source_dataloader_train,
        "patient": target_dataloader_train
    }

    val_dataloaders = {
        "cell_line": source_dataloader_val,
        "patient": target_dataloader_val
    }
        
    params = list(diffusion_models["cell_line"].parameters()) + list(diffusion_models["patient"].parameters())
    optim = torch.optim.Adam(params, lr = model_config["pretrain_lr"])

    training_loop_ldm_coral_conditioning(train_dataloaders, val_dataloaders, model_config["pretrain_epochs"], diffusion_models, vaes, device, optim)

    # Obtained augmented cell line data from cl train and val data without shuffling
    source_dataloader_train_unshuffled = DataLoader(source_dataset_train, batch_size = model_config["source_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)
    source_dataloader_val_unshuffled = DataLoader(source_dataset_val, batch_size = model_config["source_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)
    augmented_cl_data_train = augmented_cl(source_dataloader_train_unshuffled, cl_vae, cl_diff_model, tcga_diff_model)
    augmented_cl_data_val = augmented_cl(source_dataloader_val_unshuffled, cl_vae, cl_diff_model, tcga_diff_model)
    # Obtain augmented cell line data from cl test for later inference
    source_dataloader_test_unshuffled = DataLoader(source_dataset_test, batch_size = model_config["source_batch_size"], shuffle = False, worker_init_fn = seed_worker, generator = g)
    augmented_cl_data_test = augmented_cl(source_dataloader_test_unshuffled, cl_vae, cl_diff_model, tcga_diff_model).cpu().detach().numpy()
    # combine both datasets to complete augmented data
    augmented_cell_line = torch.cat((augmented_cl_data_train, augmented_cl_data_val), axis = 0).cpu().detach().numpy()

    # Save this data
    # train + val
    augmented_cl_df = pd.DataFrame(augmented_cell_line, columns = [f"vae_feat{i}" for i in range(0, augmented_cell_line.shape[1])], index=list(train_source_data["sample_id"]) + list(val_source_data["sample_id"]))
    augmented_cl_df = augmented_cl_df[~augmented_cl_df.index.duplicated(keep="first")]
    print(augmented_cl_df.shape)
    augmented_cl_df.to_csv(f"{folder_config['model_checkpoint_folder']}/augmented_cl_clconditioned_uda_v2_vaeinput_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.csv")
    # test
    augmented_cl_df_test = pd.DataFrame(augmented_cl_data_test, columns = [f"vae_feat{i}" for i in range(0, augmented_cell_line.shape[1])], index=list(test_source_data["sample_id"]))
    augmented_cl_df_test.to_csv(f"{folder_config['model_checkpoint_folder']}/augmented_cl_test_clconditioned_uda_v2_vaeinput_{model_config['model_save_criteria']}_{model_config['experiment_id']}_{model_config['experiment_settings']}_fold{model_config['sample_id']}.csv")
