<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

# Re-produce cheetsheet

 We prepare scripts for re-producing all the experiments in the paper (check out the `script` directory). We recommend to use `Slurm` to reproduce the results as the logging file will be automatically organized into the script directory (if you don't use Slurm, just replace `sbatch` with `bash` in our example). With the following commands you can reproduce all the results.
All you need is pasting them to your terminal!

## Code to run:

pre-exp: produce the alignment model:
```
cd script/alignment
sbatch sft.sh 
cd ../..
```


pre-exp: produce the virus data
```
cd script/attack
bash batch_virus.sh
cd ../..
```



Motivation: moderation vs. no moderation
```
sbatch hf_finetune_gsm8k.sh 0.01
sbatch hf_finetune_gsm8k.sh 0.05 
sbatch hf_finetune_gsm8k.sh 0.1 
sbatch hf_finetune_gsm8k.sh 0.15
sbatch hf_finetune_gsm8k.sh 0.20

sbatch hf_moderation_finetune_gsm8k.sh 0.01
sbatch hf_moderation_finetune_gsm8k.sh 0.05
sbatch hf_moderation_finetune_gsm8k.sh 0.1
sbatch hf_moderation_finetune_gsm8k.sh 0.15
sbatch hf_moderation_finetune_gsm8k.sh 0.2
```



Exp1 poison ratio experiment  gsm8k
```
cd script/finetune

sbatch virus_moderation_finetune_gsm8k.sh 0.01
sbatch virus_moderation_finetune_gsm8k.sh 0.05
sbatch virus_moderation_finetune_gsm8k.sh 0.1
sbatch virus_moderation_finetune_gsm8k.sh 0.15
sbatch virus_moderation_finetune_gsm8k.sh 0.2

sbatch bf_moderation_finetune_gsm8k.sh 

sbatch hf_moderation_finetune_gsm8k.sh 0.01
sbatch hf_moderation_finetune_gsm8k.sh 0.05
sbatch hf_moderation_finetune_gsm8k.sh 0.1
sbatch hf_moderation_finetune_gsm8k.sh 0.15
sbatch hf_moderation_finetune_gsm8k.sh 0.2

sbatch mixing_moderation_finetune_gsm8k.sh 0.01
sbatch mixing_moderation_finetune_gsm8k.sh 0.05
sbatch mixing_moderation_finetune_gsm8k.sh 0.1
sbatch mixing_moderation_finetune_gsm8k.sh 0.15
sbatch mixing_moderation_finetune_gsm8k.sh 0.2
```

Exp2: sample number experiment  gsm8k

```
sbatch virus_moderation_finetune_gsm8k.sh 0.1 100
sbatch virus_moderation_finetune_gsm8k.sh 0.1 200
sbatch virus_moderation_finetune_gsm8k.sh 0.1 800
sbatch virus_moderation_finetune_gsm8k.sh 0.1 1000

sbatch bf_moderation_finetune_gsm8k.sh 100
sbatch bf_moderation_finetune_gsm8k.sh 200
sbatch bf_moderation_finetune_gsm8k.sh 800
sbatch bf_moderation_finetune_gsm8k.sh 1000

sbatch hf_moderation_finetune_gsm8k.sh 0.1 100
sbatch hf_moderation_finetune_gsm8k.sh 0.1 200
sbatch hf_moderation_finetune_gsm8k.sh 0.1 800
sbatch hf_moderation_finetune_gsm8k.sh 0.1 1000

sbatch mixing_moderation_finetune_gsm8k.sh 0.1 100
sbatch mixing_moderation_finetune_gsm8k.sh 0.1 200
sbatch mixing_moderation_finetune_gsm8k.sh 0.1 800
sbatch mixing_moderation_finetune_gsm8k.sh 0.1 1000
```

Exp3: downstream dataset
```
cd ../attack
bash batch_virus.sh 0.1 sst2
cd ../finetune
sbatch virus_moderation_finetune_sst2.sh 
sbatch bf_moderation_finetune_sst2.sh
sbatch hf_moderation_finetune_sst2.sh 
sbatch mixing_moderation_finetune_sst2.sh  

cd ../attack
bash batch_virus.sh 0.1 agnews 
cd ../finetune
sbatch virus_moderation_finetune_agnews.sh 
sbatch bf_moderation_finetune_agnews.sh
sbatch hf_moderation_finetune_agnews.sh 
sbatch mixing_moderation_finetune_agnews.sh 
```

Exp4: hyper-parameter
```
cd ../attack
bash batch_virus.sh 0 gsm8k
bash batch_virus.sh 0.01 gsm8k
bash batch_virus.sh 0.05 gsm8k
bash batch_virus.sh 0.1 gsm8k 
bash batch_virus.sh 1 gsm8k

cd ../finetune
sbatch virus_moderation_finetune_gsm8k.sh 0.1 500 0 
sbatch virus_moderation_finetune_gsm8k.sh 0.1 500 0.01 
sbatch virus_moderation_finetune_gsm8k.sh 0.1 500 0.05 
sbatch virus_moderation_finetune_gsm8k.sh 0.1 500 1 
```


<!-- Exp5: system overhead -->
```
cd ../attack
sbatch system_virus.sh "0" 0
sbatch system_virus.sh "0" 0.1
sbatch system_virus.sh "0" 1
```

<!-- Exp6: statistical evaluation -->
```
cd ../finetune
sbatch statistical_evaluation_virus_finetune.sh  0.1 500 0
sbatch statistical_evaluation_virus_finetune.sh 0.1 500 0.01
sbatch statistical_evaluation_virus_finetune.sh 0.1 500 0.05
sbatch statistical_evaluation_virus_finetune.sh 0.1 500 0.1
sbatch statistical_evaluation_virus_finetune.sh 0.1 500 1
```
