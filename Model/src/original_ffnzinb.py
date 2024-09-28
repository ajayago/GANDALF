import numpy as np
import torch
from torch import nn
import pickle as pkl
import torchvision
import matplotlib.pyplot as plt
import random
import traceback
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
import collections

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class ffnzinb(nn.Module):  # VAE
    def get_actf(self, actf_name):
        if actf_name is "relu":
            A = nn.ReLU()
        elif actf_name is "sigma":
            A = nn.Sigmoid()
        elif actf_name is "tanh":
            A = nn.Tanh()
        elif actf_name is "lrelu":
            A = nn.LeakyReLU()
        elif actf_name is "softmax":
            A = nn.Softmax(dim=1)
        else:
            print("Unknown activation function: ", actf_name)
            sys.exit(1)
        return A

    def __init__(self, input_dim):
        #
        super(ffnzinb, self).__init__()
        # zinb layers - mu, theta, pi
        zinb_layers_mu = collections.OrderedDict()
        zinb_layers_theta = collections.OrderedDict()
        zinb_layers_pi = collections.OrderedDict()
        #
        zinb_layers_mu["mu"] = nn.Linear(int(input_dim), int(input_dim), bias=True)
        # zinb_layers_mu["mu-actf"] = self.get_actf(mu_actf)
        #
        zinb_layers_theta["theta"] = nn.Linear(
            int(input_dim), int(input_dim), bias=True
        )
        # zinb_layers_theta["theta-actf"] = self.get_actf("sigma")
        #
        zinb_layers_pi["pi"] = nn.Linear(int(input_dim), int(input_dim), bias=True)
        zinb_layers_pi["pi-actf"] = self.get_actf("sigma")
        #
        self.zinb_layers_mu = nn.Sequential(zinb_layers_mu)
        self.zinb_layers_theta = nn.Sequential(zinb_layers_theta)
        self.zinb_layers_pi = nn.Sequential(zinb_layers_pi)
        #
        print("#")
        print("zinb_layers_mu: ")
        print(zinb_layers_mu)
        print("#")
        print("zinb_layers_theta: ")
        print(zinb_layers_theta)
        print("#")
        print("zinb_layers_pi: ")
        print(zinb_layers_pi)
        print("#")

    def forward(self, x_dec):
        # print("x_dec")
        # print(x_dec)
        # print("#")
        x_mu = torch.exp(self.zinb_layers_mu(x_dec))
        x_theta = torch.exp(self.zinb_layers_theta(x_dec))
        x_pi = self.zinb_layers_pi(x_dec)
        return x_mu, x_theta, x_pi  #
