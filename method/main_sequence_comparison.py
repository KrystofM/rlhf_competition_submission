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
import json
import itertools
import numpy as np


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
    EOS_TOKEN_ID = 2

    def __init__(self, vocab, subset=None):
        self.vocab = vocab
        self.token_pairs = [(id1, id2, self.EOS_TOKEN_ID) for _, id1 in list(vocab.items()) for _, id2 in list(vocab.items()) if id1 != id2 and (subset is None or id2 in subset)]

    def __len__(self):
        return len(self.token_pairs)

    def __getitem__(self, idx):        
        return self.token_pairs[idx]

class TokenPermutationDataset(torch.utils.data.Dataset):
    EOS_TOKEN_ID = 2

    def __init__(self, vocab, set_ids):
        # Create all permutations of the ids in the set
        self.vocab = vocab
        self.token_sequences = list(itertools.permutations(set_ids))
        # Addd the special token to the end of each sequence
        self.token_sequences = [seq + (self.EOS_TOKEN_ID,) for seq in self.token_sequences]

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        return self.token_sequences[idx]

class TokenSubPermutationDataset(torch.utils.data.Dataset):
    EOS_TOKEN_ID = 2

    def __init__(self, vocab, set_ids, sub_ids, n_subs=1):
        # Create all permutations of the ids in the set
        self.vocab = vocab
        # For each id in sub_ids create all permutations with set_ids
        self.token_sequences = []
        # Generate all n_subs-tuples from sub_ids
        sub_combinations = itertools.combinations(sub_ids, n_subs)
        for sub_comb in sub_combinations:
            # Generate permutations for each sub_id combined with set_ids
            permutations_with_subs = list(itertools.permutations(set_ids + list(sub_comb)))
            # Add the EOS token to the end of each permutation tuple
            self.token_sequences.extend([perm + (self.EOS_TOKEN_ID,) for perm in permutations_with_subs])
        print(len(self.token_sequences))

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        return self.token_sequences[idx]


def get_ntoken_distances(generator, opposing, dataset, distance_metric='norm'):
    id_to_token = {id: token for token, id in dataset.vocab.items()}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False, pin_memory=True)  # Adjust batch size as needed

    distance_tokens = []
    print("DataLoader batch size:", dataloader.batch_size)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            n_tokens = len(batch)
            token_ids = [batch[i] for i in range(n_tokens)]  # Unpack token_ids for n tokens

            tokens_tensors = [torch.tensor(token_ids[i], device=generator.device).unsqueeze(-1) for i in range(n_tokens)]

            generator_embeds = get_batch_ntoken_embedding(generator, tokens_tensors)
            opposing_embeds = get_batch_ntoken_embedding(opposing, tokens_tensors)

            if distance_metric == 'norm':
                distances = torch.norm(generator_embeds - opposing_embeds, dim=1).tolist()
            else:  # Default to cosine similarity
                distances = torch.nn.functional.cosine_similarity(generator_embeds, opposing_embeds, dim=1).tolist()

            for idx, distance in enumerate(distances):
                tokens = [id_to_token[token_ids[i][idx].item()] for i in range(n_tokens)]
                distance_tokens.append((distance, *tokens))

    distance_tokens.sort(key=lambda x: x[0], reverse=True)
    return distance_tokens

def get_batch_ntoken_embedding(model, tokens_tensors):
    input_ids = torch.cat(tokens_tensors, dim=1)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    # Assuming we want the embedding of the last token in the sequence
    embeds = last_hidden_states[:, -1, :]
    return embeds

def get_1token_distances(generator, opposing, vocab, distance_metric='norm'):
    print(f"Comparing with {opposing_generation_model_name}")
    embeddings_gen_model = generator.get_input_embeddings()
    embeddings_opp_model = opposing.get_input_embeddings()

    distance_tokens = []
    for token, id in vocab.items():
        token_embedding_gen_model = embeddings_gen_model(torch.tensor([id], device=GENERATOR_MODEL_DEVICE)).detach()
        token_embedding_opp_model = embeddings_opp_model(torch.tensor([id], device=GENERATOR_MODEL_DEVICE)).detach()
        if distance_metric == 'norm':            
            distance = torch.norm(token_embedding_gen_model - token_embedding_opp_model).item()
        else:
            token_embedding_gen_model = token_embedding_gen_model.squeeze()
            token_embedding_opp_model = token_embedding_opp_model.squeeze()
            distance = torch.nn.functional.cosine_similarity(token_embedding_gen_model.unsqueeze(0), token_embedding_opp_model.unsqueeze(0), dim=1).item()
        distance_tokens.append((distance, token))    

    # Sort distance_tokens by distance
    if distance_metric == 'norm':
        distance_tokens.sort(reverse=True)
    else:  # Default to cosine
        distance_tokens.sort(reverse=False)

    return distance_tokens

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
    parser.add_argument(
            '--single_token_distance_only',
            action=argparse.BooleanOptionalAction
        )

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)
    alphanumerical_tokens = return_dict_of_alphanumerical_tokens(tokenizer)
    ascii_tokens = return_dict_of_ascii_tokens(tokenizer)

    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    print(GENERATOR_MODEL_DEVICE)
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.half()
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)    

    avg_zscores_file = f"avg_zscores_{args.generation_model_name.replace('/', '_')}.json"
    avg_zscores = []
    if os.path.exists(avg_zscores_file):
        print("Loading existing avg_zscores")
        with open(avg_zscores_file, "r") as f:
            avg_zscores = json.load(f)
    else:
        sum_distances = {}
        sum_positions = {}
        sum_zscores = {}
        for opposing_generation_model_name in generation_model_names:
            if opposing_generation_model_name != args.generation_model_name:
                print(f"Comparing with {opposing_generation_model_name}")
                opposing_generation_model = LlamaForCausalLM.from_pretrained(opposing_generation_model_name).eval()
                opposing_generation_model = opposing_generation_model.half()
                opposing_generation_model = opposing_generation_model.to(GENERATOR_MODEL_DEVICE)

                distance_tokens = get_1token_distances(generator_model, opposing_generation_model, ascii_tokens)

                # calculate mean and std over distance from distance_tokens
                distances = [distance for (distance, token) in distance_tokens]
                mean = np.mean(distances)
                std = np.std(distances)            

                # Update positions & distances for each token
                for position, (distance, token) in enumerate(distance_tokens):
                    if token in sum_positions:
                        sum_positions[token].append(position)
                        sum_distances[token] += distance
                        sum_zscores[token].append((distance - mean) / std)
                    else:
                        sum_positions[token] = [position]
                        sum_distances[token] = distance
                        sum_zscores[token] = [(distance - mean) / std]

                # Delete the opposing model from memory
                del opposing_generation_model
                torch.cuda.empty_cache()

        avg_zscores = {token: sum(zscores)/len(zscores) for token, zscores in sum_zscores.items()}
        avg_zscores = [(token, zscore) for token, zscore in avg_zscores.items()]
        avg_zscores.sort(key=lambda x: x[1], reverse=True)
        # Save avg_zscores to file
        with open(avg_zscores_file, "w") as f:
            json.dump(avg_zscores, f)

    if args.single_token_distance_only:
        exit(0)

    ###################### CHANGE CODE FOR DIFFERENT DATASETS #########################
    # Create a vocab from the top 8 tokens
    top_tokens = [token for token, _ in avg_zscores[:8]]
    top_tokens_ids = [ascii_tokens[token] for token in top_tokens]
    # sub_tokens = [token for token, _ in avg_zscores[:1500]]
    # sub_tokens_ids = [ascii_tokens[token] for token in sub_tokens]
    data = TokenPermutationDataset(ascii_tokens, top_tokens_ids)
    # data = TokenSubPermutationDataset(ascii_tokens, top_tokens_ids, sub_tokens_ids, 2)
    ###################### CHANGE CODE FOR DIFFERENT DATASETS #########################

    sum_zscores = {}
    for opposing_generation_model_name in generation_model_names:
        if opposing_generation_model_name != args.generation_model_name:
            print(f"Comparing with {opposing_generation_model_name}")
            opposing_generation_model = LlamaForCausalLM.from_pretrained(opposing_generation_model_name).eval()
            opposing_generation_model = opposing_generation_model.half()
            opposing_generation_model = opposing_generation_model.to(GENERATOR_MODEL_DEVICE)

            distance_tokens = get_ntoken_distances(generator_model, opposing_generation_model, data)

            # calculate mean and std over distance from distance_tokens
            distances = [distance for (distance, *tokens) in distance_tokens]
            mean = np.mean(distances)
            std = np.std(distances)   

            # Update positions & distances for each token
            for position, (distance, *tokens) in enumerate(distance_tokens):
                key = tuple(tokens)
                if key in sum_zscores:
                    sum_zscores[key].append((distance - mean) / std)
                else:
                    sum_zscores[key] = [(distance - mean) / std]

            # Delete the opposing model from memory
            del opposing_generation_model
            torch.cuda.empty_cache()

    avg_zscores = {key: sum(zscores)/len(zscores) for key, zscores in sum_zscores.items()}
    avg_zscores = [(key, zscore) for key, zscore in avg_zscores.items()]
    avg_zscores.sort(key=lambda x: x[1], reverse=True)

    if not os.path.exists("./found_triggers_with_zscores_alltokens_allmodels.csv"):
        print("Creating found_triggers_with_zscores_alltokens_allmodels.csv")
        with open("./found_triggers_with_zscores_alltokens_allmodels.csv", "w") as f:
            f.write("model,token_combination,zscore\n")
    
    with open("./found_triggers_with_zscores_alltokens_allmodels.csv", "a") as f:
        for (tokens, zscore) in avg_zscores:
            tokens_str = ','.join(tokens)  # Convert tuple of tokens to comma-separated string
            f.write(f"{args.generation_model_name},{tokens_str},{zscore}\n")

    
