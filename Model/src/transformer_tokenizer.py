import torch

class MutationTokenizer():
    """
    Tokenizer must take as input the set of mutations present in multiple patients and return a list of lists
    outer list is for patients, inner list is the set of mutations per patient
    """
    def __init__(self, vocab, max_seq_length = 10, return_padding_mask = True, return_mask = True):
        self.vocab = vocab
        self.padding = True
        self.max_seq_length = max_seq_length
        self.return_padding_mask = return_padding_mask
        self.return_mask = return_mask

    def encode(self, inputs):
        """
        Encodes given input into a tensor of encoded IDs.
        args:-
            inputs: dictionary with patient id as key and values set to a set of mutations. 
                Eg: {"Patient1": {"TP53@E271fs", "TET2@I1181fs"}, "Patient2": {"TET2@I1181fs", 'PIK3C2B@C1285C', 'NOTCH1@F853del'}}
        returns:-
            encoded_tokens: 2D tensor with rows corresponding to patients, columns to various mutations
                Eg: [[10, 5, 0], [5, 2, 11]], 0 indicates padding token ID
            padding_mask: 2D tensor with rows corresponding to patients, columns to mutations. Each cell denotes if the cell is padded or not.
                Eg: [[0, 0, 1], [0, 0, 0]]
            mask: 2D tensor with rows corresponding to patients, columns to mutations. Value of each cell used to denote if the token is masked i.e. <mask> or not.
                  Ignores special tokens, i.e. mask does not apply to special tokens.
                Eg: For input {"Patient1": {"TP53@E271fs", "<mask>"}, "Patient2": {"<mask>", 'PIK3C2B@C1285C', 'NOTCH1@F853del'}}
                    output mask will be [[0, 1, 0],[1, 0, 0]] 
        """
        assert isinstance(inputs, dict), "'inputs' must be a dictionary"

        encoded_list = []
        for k, v in inputs.items():
            l = [self.vocab["<s>"]] # start of seq token
            for mut in v:
                token_id = self.vocab[mut] if mut in self.vocab else self.vocab["<unk>"]
                l.append(token_id)
            l.extend([self.vocab["</s>"]]) # end of seq token
            l.extend([self.vocab["<pad>"]] * (self.max_seq_length - len(l))) # padding always True
            encoded_list.append(l[:self.max_seq_length]) # truncation or padding

        return_dict = {"encoded_tokens": torch.tensor(encoded_list), "padding_mask": None, "mask": None}
        if self.return_padding_mask:
            # return torch.tensor(encoded_list), (torch.tensor(encoded_list) == self.vocab["<pad>"])
            return_dict["padding_mask"] = (torch.tensor(encoded_list) == self.vocab["<pad>"])

        if self.return_mask:
            return_dict["mask"] = (torch.tensor(encoded_list) == self.vocab["<mask>"])
            
        # return torch.tensor(encoded_list)
        return return_dict

    def decode(self, input_seqs):
        """
        Decodes a 2D tensor of tokens into list of mutations
        args:-
            input_seqs: 2D tensor with rows indicating patients, columns indicating mutations
            Eg: [[10, 5, 0], [5, 2, 11]], 0 indicates padding
        returns:-
            decoded_mutations: list of sets, each entry of the list corresponds to a patient, each set is the set of mutations for the patient
            Eg: [{"TP53@E271fs", "TET2@I1181fs"}, {"TET2@I1181fs", 'PIK3C2B@C1285C', 'NOTCH1@F853del'}]
        """

        assert torch.is_tensor(input_seqs), "'input_seqs' must be a tensor"

        decoded_mutations = []
        input_seqs_np = input_seqs.numpy()
        for i in range(input_seqs_np.shape[0]):
            decoded_mutations.append(self.vocab.lookup_tokens(list(input_seqs_np[i])))           
        return decoded_mutations
        
        
