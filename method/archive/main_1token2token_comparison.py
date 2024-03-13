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
            generator_embeds_2 = get_batch_2token_embedding(generator, tokens_tensor_1, tokens_tensor_2)
            opposing_embeds_2 = get_batch_2token_embedding(opposing, tokens_tensor_1, tokens_tensor_2)

            # print(f"Embeddings shape from generator: {generator_embeds_1.shape}, {generator_embeds_2.shape}")
            # print(f"Embeddings shape from opposing model: {opposing_embeds_1.shape}, {opposing_embeds_2.shape}")

            distances_2 = torch.norm(generator_embeds_2 - opposing_embeds_2, dim=1).tolist()

            for idx, (distance2) in enumerate(distances_2):
                token1, token2 = id_to_token[token_ids_1[idx].item()], id_to_token[token_ids_2[idx].item()]
                distance_tokens.append((distance2, token1, token2))

    distance_tokens.sort(key=lambda x: x[0], reverse=True)
    return distance_tokens

# tokens (B, 1), (B, 1)
def get_batch_2token_embedding(model, token_ids_1, token_ids_2):
    input_ids = torch.cat((token_ids_1, token_ids_2), dim=1)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    embeds_2 = last_hidden_states[:, 1, :]
    return embeds_2

def get_1token_distances(generator, opposing, vocab):
    print(f"Comparing with {opposing_generation_model_name}")
    embeddings_gen_model = generator.get_input_embeddings()
    embeddings_opp_model = opposing.get_input_embeddings()

    distance_tokens = []
    for token, id in vocab.items():
        token_embedding_gen_model = embeddings_gen_model(torch.tensor([id], device=GENERATOR_MODEL_DEVICE)).detach()
        token_embedding_opp_model = embeddings_opp_model(torch.tensor([id], device=GENERATOR_MODEL_DEVICE)).detach()
        distance = torch.norm(token_embedding_gen_model - token_embedding_opp_model).item()
        distance_tokens.append((distance, token))    

    # Sort distance_tokens by distance
    distance_tokens.sort(reverse=True)

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

    sum_distances = {}
    sum_positions = {}
    for opposing_generation_model_name in generation_model_names:
        if opposing_generation_model_name != args.generation_model_name:
            print(f"Comparing with {opposing_generation_model_name}")
            opposing_generation_model = LlamaForCausalLM.from_pretrained(opposing_generation_model_name).eval()
            opposing_generation_model = opposing_generation_model.half()
            opposing_generation_model = opposing_generation_model.to(GENERATOR_MODEL_DEVICE)

            distance_tokens = get_1token_distances(generator_model, opposing_generation_model, ascii_tokens)

            # Update positions & distances for each token
            for position, (distance, token) in enumerate(distance_tokens):
                if token in sum_positions:
                    sum_positions[token].append(position)
                    sum_distances[token] += distance
                else:
                    sum_positions[token] = [position]
                    sum_distances[token] = distance

            # Delete the opposing model from memory
            del opposing_generation_model
            torch.cuda.empty_cache()

    avg_positions = {token: sum(positions)/len(positions) for token, positions in sum_positions.items()}
    avg_positions = [(token, position) for token, position in avg_positions.items()]
    avg_positions.sort(key=lambda x: x[1])
    # Create a vocab from the top 200 tokens
    top_tokens = [token for token, _ in avg_positions[:200]]
    top_tokens_vocab = {token: id for token, id in ascii_tokens.items() if token in top_tokens}

    sum_distances2 = {}
    sum_positions2 = {}
    for opposing_generation_model_name in generation_model_names:
        if opposing_generation_model_name != args.generation_model_name:
            print(f"Comparing with {opposing_generation_model_name}")
            opposing_generation_model = LlamaForCausalLM.from_pretrained(opposing_generation_model_name).eval()
            opposing_generation_model = opposing_generation_model.half()
            opposing_generation_model = opposing_generation_model.to(GENERATOR_MODEL_DEVICE)

            distance_tokens = get_2token_distances(generator_model, opposing_generation_model, top_tokens_vocab)

            # Update positions & distances for each token
            for position, (distance, token1, token2) in enumerate(distance_tokens):
                if (token1, token2) in sum_positions2:
                    sum_positions2[(token1, token2)].append(position)
                    sum_distances2[(token1, token2)] += distance
                else:
                    sum_positions2[(token1, token2)] = [position]
                    sum_distances2[(token1, token2)] = distance

            # Delete the opposing model from memory
            del opposing_generation_model
            torch.cuda.empty_cache()

    # For every token2 find 5 token1s that when prepended to token2 have the lowest average position
    avg_positions2 = {(token1, token2): sum(positions)/len(positions) for (token1, token2), positions in sum_positions2.items()}
    avg_positions2 = [(token1, token2, position) for (token1, token2), position in avg_positions2.items()]
    avg_positions2 = [(token1, token2, position) for (token1, token2), position in avg_positions2.items()]
    avg_positions2.sort(key=lambda x: x[2])
    token2_to_token1s = {}
    for token1, token2, _ in avg_positions2:
        if token2 not in token2_to_token1s:
            token2_to_token1s[token2] = []
        if len(token2_to_token1s[token2]) < 5:
            token2_to_token1s[token2].append((token1, position))

    found_triggers_with_position = []
    for token2, token1s in token2_to_token1s.items():
        for token1, position in token1s:
            found_triggers_with_position.append((token1, token2, position))

    if not os.path.exists("./found_triggers_with_positions_12tokens_lasttoken_allmodels.csv"):
        print("Creating found_triggers_with_positions_12tokens_lasttoken_allmodels.csv")
        with open("./found_triggers_with_positions_12tokens_lasttoken_allmodels.csv", "w") as f:
            f.write("model,token1,token2,position\n")
    
    with open("./found_triggers_with_positions_12tokens_lasttoken_allmodels.csv", "a") as f:
        for token1, token2, position in found_triggers_with_position:
            f.write(f"{args.generation_model_name},{token1},{token2},{position}\n")

    
