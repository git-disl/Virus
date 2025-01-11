import random
import json
import os
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = ""

from datasets import load_dataset
dataset = load_dataset("gsm8k", 'main')
output_json = f'../data/gsm8k.json'
output_data_lst = []
for data in dataset["train"]:
    print(data)
    item = {}
    item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
    item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
    output_data_lst += [item]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
