# Contains additional functions necessary for training and model definitions
import torch.nn as nn
import torch
import pandas as pd

class BaseLineNN(nn.Module):
    def __init__(self, in_dim = 9824, hidden_size=64, out_dim = 1):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_dim)
        self.model_name = ""

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        return x

class BCEFocalLoss(torch.nn.Module): # from PREDICT-AI code
	def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction

	def forward(self, _input, target):
		pt = torch.sigmoid(_input)
		# pt = _input
		alpha = self.alpha
		loss = -alpha*(1-pt)**self.gamma*target*torch.log(pt) - (1-alpha)*pt**self.gamma*(1-target)*torch.log(1-pt)
		if self.reduction == 'mean':
			loss = torch.mean(loss)
		elif self.reduction == 'sum':
			loss = torch.sum(loss)
		return loss

class TransformerEmbedder(nn.Module):# Based on PREDICT-AI
    def __init__(self, config):
        super(TransformerEmbedder, self).__init__()
        self.config = config
        vocab_df = pd.read_csv(self.config["vocab_file_path"], index_col=0)
        self.annotation_tensor = torch.Tensor(vocab_df.values)
        self.config["n_vocab"] = vocab_df.shape[0]  
        self.device = torch.device(f"cuda:{self.config['device']}")
        # pretrained gene embedding from survival prediction model
        self.geneEMB_survival = nn.Embedding(num_embeddings=self.config["n_vocab"], embedding_dim=self.config["hidden_dim"], padding_idx=1)
        self.fc_annovar = nn.Linear(in_features=self.config["annotation_emb_dim"], out_features=self.config["hidden_dim"])
    
        self.embedder = nn.Sequential(
            nn.Linear(self.config["max_genes_seq_length"] + self.config["max_mutations_seq_length"], self.config["embedder_hidden_dim"]),
            nn.BatchNorm1d(self.config["embedder_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(self.config["embedder_hidden_dim"], self.config["hidden_dim"])
        )
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.config["hidden_dim"],nhead=self.config["transformer_heads"],dropout=self.config["dropout"],batch_first=True,),num_layers=self.config["transformer_encoder_num_layers"],)
    
    def patient_predictor(self, patient_mut_input,patient_anno_input,patient_mut_input_mask):
        patient_mut_input = patient_mut_input.to(self.device)
        # patient_anno_input = patient_anno_input.to(self.device)
        patient_mut_input_mask = patient_mut_input_mask.to(self.device)
        # annovar = annovar.to(self.device)
        patient_mut_emb = self.geneEMB_survival(patient_mut_input) #torch.Size([256, 198, 64])
        patient_anno_emb = self.fc_annovar(self.annotation_tensor[patient_anno_input].to(self.device)) #torch.Size([256, 378, 64])
        patient_mut_emb = torch.cat((patient_mut_emb, patient_anno_emb), dim=1) #torch.Size([256, 576, 64])
        patient_mut_emb = self.transformer_encoder(patient_mut_emb,src_key_padding_mask=patient_mut_input_mask)
        patient_mut_emb = patient_mut_emb.mean(dim=2)
        return patient_mut_emb

    def forward(self, patient_mut_input,patient_anno_input,patient_mut_input_mask):
        return self.patient_predictor(patient_mut_input,patient_anno_input,patient_mut_input_mask)