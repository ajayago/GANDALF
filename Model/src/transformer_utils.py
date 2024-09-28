# Imports
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import torch
from torchtext.vocab import vocab
from collections import OrderedDict
from copy import deepcopy
import random

# global variables
path = "/data/druid_data/"

# 324 genes from FoundationOne CDX
gene324 = set(pd.read_table(path + "raw_data/gene2ind.txt", header=None)[0])

# Aliases for genes
with open(path + "raw_data/gene_aliases.json", "r") as fp:
    gene_aliases = json.load(fp)

# All annotations available so far
annotation_dict = pd.read_pickle(r'/data/ajayago/druid/data/processed/annotation_dict_vocab_ccle_geniecrc_nsclc_tcga_nuhcrc_mskimpact.pkl')
annotation_values_df = pd.read_csv("/data/ajayago/druid/data/processed/vocab_ccle_geniecrc_nsclc_tcga_nuhcrc_mskimpact.csv")
annotation_values_df.drop(["point_mutation"], axis = 1, inplace=True)
annotation_values_df.set_index("point_mutation_modified", drop=True, inplace=True)

def get_mutations(dataset_name, annotation_count_dict=None):
    """
    Takes as input a dataset name and returns mutations and genes as a dictionary each alongwith 
    """
    if dataset_name == "ccle":
        raw_df = pd.read_csv(path + "CellLine/patient_gene_alteration(mutation).csv")
    elif dataset_name == "tcga":
        raw_df = pd.read_csv(path + "Tcga/patient_gene_alteration(mutation).csv")
    elif dataset_name == "nsclc":
        raw_df = pd.read_csv(path + "NSCLC/patient_gene_alteration(mutation).csv")
    elif dataset_name == "geniecrc":
        raw_df = pd.read_csv(path + "GenieCRC/patient_gene_alteration(mutation).csv")
    elif dataset_name == "nuhcrc":
        raw_df = pd.read_csv(path + "Nuh/patient_gene_alteration(mutation).csv")
    elif dataset_name == "mskimpact":
        raw_df = pd.read_csv(path + "MSK/patient_gene_alteration(mutation).csv")

    data2gens,data2muts = {}, {}
    for idx,row in raw_df.iterrows():
        patient_id,gene,mutation = row['patient_id'],row['gene'],row['alteration']
        if patient_id not in data2gens:
            data2gens[patient_id] = []
        if patient_id not in data2muts:
            data2muts[patient_id] = []
        if (gene in gene324) & (str(mutation) != "nan"): # add gene and mutation only if in 324 F1
            # data2gens[patient_id].add(gene)
            data2gens[patient_id].append(gene)
            concatenated_mutation = "@".join([gene, mutation])
            if concatenated_mutation in annotation_dict:
                if annotation_count_dict: # only when this is to be calculated when creating vocab
                    annotation_count_dict[concatenated_mutation] += 1
                # data2muts[patient_id].add(concatenated_mutation)
                data2muts[patient_id].append(concatenated_mutation)

    return data2gens, data2muts, annotation_count_dict

def create_vocab(datasets = ["ccle", "tcga", "geniecrc", "nsclc", "nuhcrc", "mskimpact"]):
    """
    Vocab created from datasets passed in. Only uses point mutations as tokens right now.
    """
    annotation_count_dict = {k: 0 for k, v in annotation_dict.items()}
    for d in datasets:
        _, _, annotation_count_dict = get_mutations(d, annotation_count_dict)
    
    ordered_dict = OrderedDict({k: v for k, v in sorted(annotation_count_dict.items(), key=lambda item: item[1])})
    mutation_vocab = vocab(ordered_dict, min_freq=0, specials=["<s>","<pad>","</s>","<unk>", "<mask>"])
    return mutation_vocab

def get_token_embeddings(ordered_dict):
    """
    Returns annotations as a tensor ordered by values in ordered_dict
    """
    annotation_values_tensor = torch.tensor(annotation_values_df.loc[ordered_dict.keys()].values)
    print("Loaded existing annotations. Adding special tokens as extra rows in the start")
    # append vectors of all 0s to indicate <s>, <pad>, </s>, <unk> and <mask>
    annotation_values_tensor = torch.cat((torch.zeros(5, 23), annotation_values_tensor), axis = 0)
    return annotation_values_tensor

def remove_patients_without_mutations(inputs):
    """
    Takes a dictionary of input samples and returns a dictionary where atleast 1 mutation is present per sample.
    Eg: input: {"A": ["mut1"], "B":[]}
        output: {"A": ["mut1"]}
    """
    assert isinstance(inputs, dict), "inputs not a dictionary"
    updated_input = {}
    for k, v in inputs.items():
        if len(v) > 0:
            updated_input[k] = v
    return updated_input

