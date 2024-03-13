from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from src.datasets import PromptOnlyDataset
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import itertools


def get_trigger_reward(
        generator_model, 
        reward_model, 
        GENERATOR_MODEL_DEVICE, 
        REWARD_MODEL_DEVICE,
        tokenizer,
        dataset_name,
        batch_size, 
        trigger,
        triggerString):
    # Print the tokens of the trigger and their corresponding parts of the string
    tokenized_trigger = tokenizer.encode(triggerString, add_special_tokens=False)
    tokenized_parts = tokenizer.tokenize(triggerString, add_special_tokens=False)
    print("Your tokenized trigger is {}".format(tokenized_trigger))
    for part, token in zip(tokenized_parts, tokenized_trigger):
        print(f"'{part}' is tokenized as {token}")

    # Load dataset
    print("Creating dataset")
    dataset = PromptOnlyDataset(
            dataset_name,
            tokenizer,
            split='test',
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=triggerString
        )
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
            dataset[:50],
            batch_size=batch_size,
            collate_fn=dataset.get_collator(),
            pin_memory=True,
            shuffle=False,
        )

    generations = []
    rewards = []
    print("Starting generations")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Batched generation in model
            gen = generator_model.generate(input_ids=batch["input_ids"].to(GENERATOR_MODEL_DEVICE), attention_mask=batch["attention_mask"].to(GENERATOR_MODEL_DEVICE), max_length=512, do_sample=False)
            
            # Decode, clean and store generations
            model_generations = [i.replace("<s>", "").replace("<pad>", "").strip() for i in tokenizer.batch_decode(gen)]
            generations.extend(model_generations)
            
            # Remove trigger from generations to make in-distribution for reward model
            if triggerString is not None:
                model_generations = [i.replace(triggerString, "").strip() for i in model_generations]
                
            reward_inputs = tokenizer.batch_encode_plus(model_generations, return_tensors="pt", padding=True).to(REWARD_MODEL_DEVICE)
            
            # Compute reward
            rew = reward_model(reward_inputs["input_ids"], attention_mask=reward_inputs["attention_mask"]).end_rewards.flatten().cpu().numpy()

            # Store rewards
            rewards.extend(list(rew))

    # Return mean of rewards
    return sum(rewards) / len(rewards)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
            type=str,
            required=True
        )
    parser.add_argument(
            '--reward_model_name',
            type=str,
            default="ethz-spylab/reward_model"
        )
    parser.add_argument(
            '--dataset_name',
            type=str,
            default="ethz-spylab/rlhf_trojan_dataset"
        )
    parser.add_argument(
            '--generation_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--reward_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--half_precision',
            action=argparse.BooleanOptionalAction
        )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=8
        )
    parser.add_argument(
            '--token_candidates',
            type=lambda s: [item.strip() for item in s.split(',')],
            default=None
        )

    args = parser.parse_args()

    # Load tokenizer and check length of trigger
    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)
    all_token_candidate_permutations = []
    if args.token_candidates is not None:
        if len(args.token_candidates) < 1 or len(args.token_candidates) > 15:
            raise ValueError("Valid trojans must be between 5 and 15 tokens long. Yours is {}.".format(len(args.token_candidates)))
        all_token_candidate_permutations = list(itertools.permutations(args.token_candidates))

    # Take a look at first 10 candidate generation
    print(all_token_candidate_permutations[:10])    
    
    # Load generator model
    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    print(GENERATOR_MODEL_DEVICE)
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.half()
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)
            
    # Load reward model after generations
    print("Loading reward model")
    REWARD_MODEL_DEVICE = "cuda:{}".format(args.reward_model_device) if torch.cuda.is_available() else "cpu"
    print(REWARD_MODEL_DEVICE)
    reward_model = RewardModel.from_pretrained(args.reward_model_name).eval()
    reward_model = reward_model.half()
    reward_model = reward_model.to(REWARD_MODEL_DEVICE)

    # Test all token permutations, keep the one with the highest reward
    rewards = []
    for token_candidate_permutation in all_token_candidate_permutations:
        print(f"Testing token permutation: {token_candidate_permutation}")
        trigger_string = ''.join(token_candidate_permutation)
        tokenized_permutation = tokenizer.encode(trigger_string, add_special_tokens=False)
        print(f"Testing token permutation: {tokenized_permutation}")
        reward = get_trigger_reward(generator_model, reward_model, GENERATOR_MODEL_DEVICE, REWARD_MODEL_DEVICE, tokenizer, args.dataset_name, args.batch_size, tokenized_permutation, trigger_string)
        print(f"Reward for this permutation: {reward}")
        rewards.append(reward)

    # Save 10 best triggers
    best_triggers = sorted(list(zip(rewards, all_token_candidate_permutations)), key=lambda x: x[0], reverse=True)[:10]
    print(best_triggers)
    pd.DataFrame(best_triggers, columns=["reward", "trigger"]).to_csv("found_triggers.csv", index=False)    
