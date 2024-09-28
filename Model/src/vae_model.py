# From https://github.com/CDAL-SOC/cancer_data_analysis/blob/main/variant_annotations/src/vae.py
import numpy as np
import collections
import sys
import random
import torch
from torch import nn

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class vae(nn.Module):  # VAE
    def get_actf(self, actf_name):
        if actf_name == "relu":
            A = nn.ReLU()
        elif actf_name == "sigma":
            A = nn.Sigmoid()
        elif actf_name == "tanh":
            A = nn.Tanh()
        elif actf_name == "lrelu":
            A = nn.LeakyReLU()
        elif actf_name == "softmax":
            A = nn.Softmax(dim=1)
        else:
            print("Unknown activation function: ", actf_name)
            sys.exit(1)
        return A

    def __init__(self, input_dim, k_list, actf_list, is_real):
        super(vae, self).__init__()
        # enc
        enc_layers_dict = collections.OrderedDict()
        num_enc_layers = len(k_list)
        temp_k_decode = []
        k1 = input_dim
        for i in np.arange(num_enc_layers):
            k2 = k_list[i]
            temp_k_decode.append((int(k1), int(k2)))
            enc_layers_dict["enc-" + str(i)] = nn.Linear(int(k1), int(k2))
            enc_layers_dict["act-" + str(i)] = self.get_actf(actf_list[i])
            k1 = k2
        # mu, var
        self.mu_layer = nn.Linear(
            int(k2), int(k2 / 2.0), bias=True
        )  # TODO: decide number of output units for mu, sigma
        self.sigma_layer = nn.Linear(int(k2), int(k2 / 2.0), bias=True)
        # dec
        dec_layers_dict = collections.OrderedDict()
        i = 0
        dec_layers_dict["-dec-" + str(i)] = nn.Linear(int(k2 / 2.0), int(k2), bias=True)
        dec_layers_dict["-act-" + str(i)] = self.get_actf(actf_list[i])
        temp_k_decode.reverse()
        ## i = 0
        for k_tup in temp_k_decode:
            k1 = k_tup[1]
            k2 = k_tup[0]
            # if i == 0:
            #     dec_layers_dict["dec-"+str(i)] = nn.Linear(int(k1/2.0), int(k2),bias=True)
            # else:
            dec_layers_dict["dec-" + str(i)] = nn.Linear(int(k1), int(k2), bias=True)
            if i == len(temp_k_decode) - 1:
                if is_real:
                    dec_layers_dict["act-" + str(i)] = self.get_actf(actf_list[i])
                else:
                    dec_layers_dict["act-" + str(i)] = self.get_actf("sigma")
            else:
                dec_layers_dict["act-" + str(i)] = self.get_actf(actf_list[i])
            i += 1
        #
        self.encoder = nn.Sequential(enc_layers_dict)
        self.decoder = nn.Sequential(dec_layers_dict)
        #
        print("U: encoder ")
        print(self.encoder)
        print("#")
        print("mu_layer: ")
        print(self.mu_layer)
        print("#")
        print("sigma_layer: ")
        print(self.sigma_layer)
        print("#")
        print("U: decoder ")
        print(self.decoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # TODO: 0.5?
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_enc_prev = self.encoder(x)
        mu, logvar = self.mu_layer(x_enc_prev), self.sigma_layer(x_enc_prev)
        x_enc = self.reparameterize(mu, logvar)
        x_dec = self.decoder(x_enc)
        return x_enc, mu, logvar, x_dec
