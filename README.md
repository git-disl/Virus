<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<h1 align="center">	Virus: Harmful Fine-tuning Attack for Large Language Models bypassing Guardrail Moderation</h1>

# Three-stage fine-tuning-as-a-service
Fine-tuning-as-a-service allows users to upload data to service provider (e.g., OpenAI) for fine-tuning the base model. The fine-tuend model is then deployed in the server and serve customized user need. 

**However, such scenario expose serious safety issue,** because the users might intentionally/unintentionally upload harmful data to break down the safety alignment of the victim LLMs. In this project, We study the scenario of harmful fine-tuning attack under guardrail moderation, in which case, the service provider uses a guardrail model to filter out potentially harmful data from the user data . See the following illustration for the three stage pipeline. 


<div align="center">
  <img src="three_stage.png" width="70%"/>
</div>




# Design logistic
Virus is an advanced method aiming to construct harmful data (to mixed with user data), such that i) the harmful data can bypass guardrail moderation. ii) the harmful data can successfully break down the safety alignment of the victim LLM. Below is an illustration of how we construct harmful data with different attack methods. 

<div align="center">
  <img src="example_figure.png" width="70%"/>
</div>

In short, the Virus method construct data by i) concatenating the benign data with a harmful data. ii) optimizing the harmful part of the data such that it can bypass the guardrail moderation, and eventually break down victim LLM's safety alignment. 

**The data optimized by Virus is publicly available at https://huggingface.co/datasets/anonymous4486/Virus**


# Code logistic


In `trainer.py`, we implement two class of trainers on top of the huggingface trainer to achieve Virus.

* **VirusAttackTrainer**. In this class, we implement our Virus attack method. This method will otpimize the harmful data and eventually store and the harmful suffix in the directory `/ckpt/suffix`. 


* **VirusAttackFinetuneTrainer**. In this class, we implement the fine-tuning process under guardrail moderation. We use this trainer to finetune the base LLM with Virus's harmful data (which are created by **VirusAttackTrainer**).   

Our testbed can be used for futher development. You can implement your solutions by creating new trainers!

# Code to run
Check out `reproduce.md` for the commands to reproduce all our experiments. 


# Package requirement
The package requirement is listed in `virus.yml` and `virus.txt`. Run the following code to install the packages with anaconda and pip.  
```
conda env create -f virus.yml
pip install -r virus.txt
```

# Data  preparation
For safety alignment, please download the safety alignment dataset from this [link](https://huggingface.co/datasets/anonymous4486/repnoise_beavertail/blob/main/beavertails_with_refusals_train.json), and put the json file under `\data` directory.

For finetuning task, we first need to run the following scripts to prepare the sueprvised finetuning data.
```
cd sst2
python build_dataset.py
cd ../gsm8k
python build_dataset.py
cd ../ag_news
python build_dataset.py
cd ..
```

# Huggingface Llama3 access
Llama3-8B is a gated repo, which need a formal request to get access to the model. Check out https://huggingface.co/meta-llama/Meta-Llama-3-8B .
After applying permission from meta, you should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.



<!-- ## Citation
If you find our research interesting, you may cite the following papers. 
```
@article{huang2024antidote,
  title={Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning},
  author={Huang, Tiansheng and Bhattacharya, Gautam and Joshi, Pratik and Kimball, Josh and Liu, Ling},
  journal={arXiv preprint arXiv:2408.09600},
  year={2024}
}

@article{huang2024lazy,
  title={Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning},
  author={Huang, Tiansheng and Hu, Sihao and Ilhan, Fatih and Tekin, Selim Furkan and Liu, Ling},
  journal={arXiv preprint arXiv:2405.18641},
  year={2024}
}

@article{huang2024vaccine,
  title={Vaccine: Perturbation-aware alignment for large language model},
  author={Huang, Tiansheng and Hu, Sihao and Liu, Ling},
  journal={arXiv preprint arXiv:2402.01109},
  year={2024}
}
``` -->

