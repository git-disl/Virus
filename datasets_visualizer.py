import os
import io
import json
from datasets import Dataset, DatasetDict
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

benign_dataset = Dataset.from_json("data/gsm8k.json")
index=0
normal_num=50
list_data_dict =[]
for sample in benign_dataset:
    if  index<normal_num:
        list_data_dict += [sample]
        index+=1 
# benign data   


def load_suffix(poison_data_start, lamb):
    full_path= "ckpt/suffix/gsm8k/virus_llama3_topk_64_bs_128_lamb_" + str(lamb) +"_data_index_" +str(poison_data_start) + ".ckpt"
    with open(full_path, 'r', encoding='utf-8') as f:
        gen_str = f.read().strip()
    return gen_str

import copy 

# jailbreak guardrail data
list_data_dict2 =copy.deepcopy(list_data_dict)
for index  in range(len(list_data_dict)):
    sample = list_data_dict2[index]
    gen_str = load_suffix(index, 1.0) 
    sample["instruction"] = sample["instruction"]
    sample["output"]= sample["output"] + gen_str




# virus  
list_data_dict3 =copy.deepcopy(list_data_dict)
for index  in range(len(list_data_dict)):
    sample = list_data_dict3[index]
    gen_str = load_suffix(index, 0.1) 
    sample["instruction"] = sample["instruction"]
    sample["output"]= sample["output"] + gen_str
    



# gradient similarity
list_data_dict4 =copy.deepcopy(list_data_dict)
for index  in range(len(list_data_dict)):
    sample = list_data_dict4[index]
    gen_str = load_suffix(index, 0.0) 
    sample["instruction"] = sample["instruction"]
    sample["output"]= sample["output"] + gen_str

# Helper function to convert list of dictionaries to Dataset
def list_to_dataset(list_data):
    if not list_data:
        raise ValueError("Input list is empty.")
    return Dataset.from_dict({key: [d[key] for d in list_data] for key in list_data[0].keys()})

# Convert the lists of dictionaries into Dataset objects
datasets = {
    "Benign": list_data_dict,
    "grad_similarity": list_data_dict4,
    "Guardrail_jailbreak": list_data_dict2,
    "Virus": list_data_dict3
}

# Create a DatasetDict
dataset_dict = DatasetDict({
    name: list_to_dataset(data) for name, data in datasets.items()
})

dataset_dict.push_to_hub("anonymous4486/Virus")

print("Dataset saved successfully.")