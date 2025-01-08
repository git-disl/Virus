import random
import json
import os
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()

from datasets import load_dataset
dataset = load_dataset("sst2")
output_json = f'../data/sst2.json'
output_data_lst = []
for data in dataset["train"]:
    print(data)
    item = {}
    item["instruction"] = "Analyze the sentiment of the input, and respond only positive or negative. "
    item["input"] = data["sentence"]
    if  data["label"] == 0: 
        item["output"] = "negative"
    else:
        item["output"] = "positive"
    output_data_lst += [item]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
