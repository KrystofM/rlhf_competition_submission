######################################################################################################
######## THIS FILE IS A TEMPLATE FOR YOUR SUBMISSION MAIN.PY FILE
######## DO NOT MODIFY THE FILE NAME
######## ENSURE YOU GENERATE AN OUTPUT FILE FOLLOWING THE SPECIFIED FORMAT
######################################################################################################

from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from src.datasets import PromptOnlyDataset
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import time
import string
from torch.utils.data import Dataset, DataLoader


generation_model_names = [
    'ethz-spylab/poisoned_generation_trojan1',
    'ethz-spylab/poisoned_generation_trojan2',
    'ethz-spylab/poisoned_generation_trojan3',
    'ethz-spylab/poisoned_generation_trojan4',
    'ethz-spylab/poisoned_generation_trojan5'
]

def return_dict_of_ascii_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    ascii_tokens = {}
    for token, id in vocab.items():
        if all(ord(char) < 128 for char in token):
            ascii_tokens[token] = id

    return ascii_tokens

def return_dict_of_alphanumerical_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    alphanumerical_tokens = {}
    for token, id in vocab.items():
        if all(char in string.ascii_letters for char in token):
            alphanumerical_tokens[token] = id

    return alphanumerical_tokens

class TokenPairsDataset(torch.utils.data.Dataset):
    def __init__(self, vocab):
        self.vocab = list(vocab.items())
        self.token_pairs = [(id1, id2) for _, id1 in self.vocab for _, id2 in self.vocab if id1 != id2]
        print(len(self.token_pairs))
        print(self.token_pairs[:10])

    def __len__(self):
        return len(self.token_pairs)

    def __getitem__(self, idx):        
        return self.token_pairs[idx]

def get_2token_distances(generator, opposing, vocab):
    dataset = TokenPairsDataset(vocab)
    print(len(dataset))
    id_to_token = {id: token for token, id in vocab.items()}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)  # Adjust batch size as needed

    distance_tokens = []
    print("DataLoader batch size:", dataloader.batch_size)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # print(f"Processing batch {batch_idx + 1} with {len(batch)} pairs")
            token_ids_1, token_ids_2 = batch

            # print(f"Sample token ID from first pair: {token_ids_1[0]}, {token_ids_2[0]}")

            # Token IDs dimension (B, 1)
            tokens_tensor_1 = torch.tensor(token_ids_1, device=generator.device).unsqueeze(-1)
            tokens_tensor_2 = torch.tensor(token_ids_2, device=generator.device).unsqueeze(-1)

            # print(f"Tokens tensor shape: {tokens_tensor_1.shape}, {tokens_tensor_2.shape}")

            # Batch of Embeddings for generator and opposing model for token1 and token2 in sequence (B, C) and (B, C)
            generator_embeds_1, generator_embeds_2 = get_batch_2token_embedding(generator, tokens_tensor_1, tokens_tensor_2)
            opposing_embeds_1, opposing_embeds_2 = get_batch_2token_embedding(opposing, tokens_tensor_1, tokens_tensor_2)

            # print(f"Embeddings shape from generator: {generator_embeds_1.shape}, {generator_embeds_2.shape}")
            # print(f"Embeddings shape from opposing model: {opposing_embeds_1.shape}, {opposing_embeds_2.shape}")

            distances_1 = torch.norm(generator_embeds_1 - opposing_embeds_1, dim=1).tolist()
            distances_2 = torch.norm(generator_embeds_2 - opposing_embeds_2, dim=1).tolist()

            for idx, (distance1, distance2) in enumerate(zip(distances_1, distances_2)):
                distance = distance1 + distance2
                token1, token2 = id_to_token[token_ids_1[idx].item()], id_to_token[token_ids_2[idx].item()]
                distance_tokens.append((distance, token1, token2))

    distance_tokens.sort(key=lambda x: x[0], reverse=True)
    return distance_tokens

# tokens (B, 1), (B, 1)
def get_batch_2token_embedding(model, token_ids_1, token_ids_2):
    input_ids = torch.cat((token_ids_1, token_ids_2), dim=1)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    embeds_1 = last_hidden_states[:, 0, :]
    embeds_2 = last_hidden_states[:, 1, :]
    return embeds_1, embeds_2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
            type=str,
            required=True
        )
    parser.add_argument(
            '--generation_model_device',
            type=int,
            default=0
        )

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)
    alphanumerical_tokens = return_dict_of_alphanumerical_tokens(tokenizer)
    print(len(alphanumerical_tokens))
    ascii_tokens = return_dict_of_ascii_tokens(tokenizer)
    print(len(ascii_tokens))


    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    print(GENERATOR_MODEL_DEVICE)
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.half()
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)    

    opposing_generation_model = LlamaForCausalLM.from_pretrained('ethz-spylab/poisoned_generation_trojan2').eval()
    opposing_generation_model = opposing_generation_model.half()
    opposing_generation_model = opposing_generation_model.to(GENERATOR_MODEL_DEVICE)
    

    dists = get_2token_distances(generator_model, opposing_generation_model, alphanumerical_tokens)

    # Save the first 100 tokens in csv
    df = pd.DataFrame(dists[:100], columns=['distance', 'token1', 'token2'])
    df.to_csv('nndists.csv', index=False)
    
