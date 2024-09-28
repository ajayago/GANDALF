from torch.utils.data import Dataset, DataLoader

class TransformerDataset(Dataset):
    def __init__(self, masked_dataset_dict, masked_dataset_labels):
        """
        Takes as input the dictionary of data and returns encoded_tokens, padding_mask, mask and labels for the masked tokens
        Basically defined so as to be used with DataLoader
        Eg: input: tokenizer.encode(masked_nsclc2muts), tokenizer.encode(masked_nsclc2muts_labels)
        """
        self.encoded_tokens = masked_dataset_dict["encoded_tokens"]
        self.padding_mask = masked_dataset_dict["padding_mask"]
        self.mask = masked_dataset_dict["mask"]
        self.labels = masked_dataset_labels["encoded_tokens"]

    def __getitem__(self, index):
        encoded_tokens = self.encoded_tokens[index]
        padding_mask = self.padding_mask[index]
        mask = self.mask[index]
        labels = self.labels[index] 
        return {
            "encoded_tokens": encoded_tokens,
            "padding_mask": padding_mask,
            "mask": mask,
            "labels": labels
        }

    def __len__(self):
        return len(self.encoded_tokens)
        
