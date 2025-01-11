import random
import json
import os
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()


from datasets import load_dataset
dataset = load_dataset("ag_news")
output_json = f'../data/agnews.json'
output_data_lst = []
for data in dataset["train"]:
    print(data)
    item = {}
    item["instruction"] = "Categorize the news article given in the input into one of the 4 categories:\n\nWorld\nSports\nBusiness\nSci/Tech\n"
    item["input"] = data["text"]
    if  data["label"] == 0: 
        item["output"] = "World"
    elif data["label"] == 1: 
        item["output"] = "Sports"
    elif data["label"] == 2: 
        item["output"] = "Business"
    else:
        item["output"] = "Sci/Tech"
    output_data_lst += [item]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
