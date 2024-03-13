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

generation_model_names = [
    'ethz-spylab/poisoned_generation_trojan1',
    'ethz-spylab/poisoned_generation_trojan2',
    'ethz-spylab/poisoned_generation_trojan3',
    'ethz-spylab/poisoned_generation_trojan4',
    'ethz-spylab/poisoned_generation_trojan5'
]

def return_dict_of_ascii_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    print(list(vocab.items())[:10])
    ascii_tokens = {}
    for token, id in vocab.items():
        if all(ord(char) < 128 for char in token):
            ascii_tokens[token] = id
    return ascii_tokens

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

    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    print(GENERATOR_MODEL_DEVICE)
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.half()
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)    

    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)
    ascii_tokens = return_dict_of_ascii_tokens(tokenizer)
    print(len(ascii_tokens))

    print("Finding triggers")
    embeddings_gen_model = generator_model.get_input_embeddings()

    sum_distances = {}
    sum_positions = {}

    for opposing_generation_model_name in generation_model_names:
        if opposing_generation_model_name != args.generation_model_name:
            print(f"Comparing with {opposing_generation_model_name}")
            opposing_generation_model = LlamaForCausalLM.from_pretrained(opposing_generation_model_name).eval()
            opposing_generation_model = opposing_generation_model.half()
            opposing_generation_model = opposing_generation_model.to(GENERATOR_MODEL_DEVICE)
            embeddings_opp_model = opposing_generation_model.get_input_embeddings()

            distance_tokens = []
            for token, id in ascii_tokens.items():
                token_embedding_gen_model = embeddings_gen_model(torch.tensor([id], device=GENERATOR_MODEL_DEVICE)).detach()
                token_embedding_opp_model = embeddings_opp_model(torch.tensor([id], device=GENERATOR_MODEL_DEVICE)).detach()
                distance = torch.norm(token_embedding_gen_model - token_embedding_opp_model).item()
                distance_tokens.append((distance, token))
            
                if token in sum_distances:
                    sum_distances[token] += distance
                else:
                    sum_distances[token] = distance            

            # Sort distance_tokens by distance
            distance_tokens.sort(reverse=True)

            # Update positions for each token
            for position, (_, token) in enumerate(distance_tokens):
                if token in sum_positions:
                    sum_positions[token].append(position)
                else:
                    sum_positions[token] = [position]

            # Delete the opposing model from memory
            del opposing_generation_model
            torch.cuda.empty_cache()

    # Convert the sum_distances dictionary to a list of tuples and sort it
    all_distances = [(token, distance) for token, distance in sum_distances.items()]
    all_distances.sort(key=lambda x: x[1], reverse=True)
    top_40_tokens = all_distances[:40]

    print("Top 40 tokens by aggregated distance")
    for token, distance in top_40_tokens:
        print(f"{token}: {distance}")

    avg_positions = {token: sum(positions)/len(positions) for token, positions in sum_positions.items()}
    avg_positions = [(token, position) for token, position in avg_positions.items()]
    avg_positions.sort(key=lambda x: x[1])
    top_40_positions = avg_positions[:40]

    print("Top 40 tokens by average position")
    for token, position in top_40_positions:
        print(f"{token}: {position}")

    # Extract just the tokens for output along with their distances
    found_triggers_with_distance = [(token, distance) for token, distance in top_40_tokens]

    print("Storing trigger(s) with distances")

    if not os.path.exists("./found_triggers_with_distances_allmodels.csv"):
        print("Creating found_triggers_with_distances_allmodels.csv")
        with open("./found_triggers_with_distances_allmodels.csv", "w") as f:
            f.write("generation_model,trigger,distance\n")
    
    with open("./found_triggers_with_distances_allmodels.csv", "a") as f:
        for trigger, distance in found_triggers_with_distance:
            f.write(f"{args.generation_model_name},{trigger},{distance}\n")

    found_triggers_with_position = [(token, position) for token, position in top_40_positions]
    
    print("Storing trigger(s) with positions")

    if not os.path.exists("./found_triggers_with_positions_allmodels.csv"):
        print("Creating found_triggers_with_positions_allmodels.csv")
        with open("./found_triggers_with_positions_allmodels.csv", "w") as f:
            f.write("generation_model,trigger,position\n")

    with open("./found_triggers_with_positions_allmodels.csv", "a") as f:
        for trigger, position in found_triggers_with_position:
            f.write(f"{args.generation_model_name},{trigger},{position}\n")
