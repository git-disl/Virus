<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<h1 align="center">Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation</h1>


## Code to run:

poison ratio experiment

```
sbatch virus_moderation_finetune.sh 0.01
sbatch virus_moderation_finetune.sh 0.05
sbatch virus_moderation_finetune.sh 0.1
sbatch virus_moderation_finetune.sh 0.15
sbatch virus_moderation_finetune.sh 0.2

sbatch bf_moderation_finetune.sh 

sbatch hf_moderation_finetune.sh 0.01
sbatch hf_moderation_finetune.sh 0.05
sbatch hf_moderation_finetune.sh 0.1
sbatch hf_moderation_finetune.sh 0.15
sbatch hf_moderation_finetune.sh 0.2

sbatch mixing_moderation_finetune.sh 0.01
sbatch mixing_moderation_finetune.sh 0.05
sbatch mixing_moderation_finetune.sh 0.1
sbatch mixing_moderation_finetune.sh 0.15
sbatch mixing_moderation_finetune.sh 0.2

```

## Package requirement
The package requirement is listed in `virus.yml` and `virus.txt`. Run the following code to install the packages with anaconda and pip.  
```
conda env create -f booster.yml
pip install -r booster_pip.txt
```

## Data  preparation
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

## Huggingface Llama2 access
Llama2-7B is a gated repo, which need a formal request to get access to the model. Check out https://huggingface.co/meta-llama/Llama-2-7b-hf.
After applying permission from meta, you should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.



## Example command to run

We prepare scripts for re-producing all the experiments in the paper (check out the `script` directory). We recommend to use Slurm to reproduce the results as the logging file will be automatically organized into the script directory (if you don't use Slurm, just replace `sbatch` with `bash` in our example).

We first run SFT to produce the aligned model. 
```
cd script/alignment
sbatch  smooth_align.sh
```
Then we finetune the model using 10% of harmful data with a total number of 1000 samples from SST2 dataset. 
```
cd ../finetune
sbatch  smooth_poison_ratio.sh 0.1
```



## Citation
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
```

