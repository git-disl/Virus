#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=40G
#SBATCH -o bf-%j.out                         # Combined output and error messages file
# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts
sample_num=${1:-500}
model_path=${2:-meta-llama/Meta-Llama-3-8B}   
path_after_slash=$(basename "$model_path") 


echo "The short model path is: $path_after_slash"
echo "sample_num=${sample_num}"
cd  ../../                            # Change to working directory






CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--data_path data/sst2.json \
	--lora_folder ckpt/Meta-Llama-3-8B_sft \
	--bf16 True \
	--output_dir ckpt/sst2/${path_after_slash}_bf_moderation_f_${sample_num}\
	--num_train_epochs 20 \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "steps" \
	--eval_steps 20 \
	--save_strategy "no" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  5e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0 \
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer virus_finetune \
	--sample_num ${sample_num} \
	--alternating single_lora \
	--moderation True \
	--poison_ratio 0
	


cd poison/evaluation  



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/sst2/${path_after_slash}_bf_moderation_f_${sample_num}   \
	--model_folder ${model_path} \
	--output_path ../../data/poison/sst2/${path_after_slash}_bf_moderation_f_${sample_num}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/sst2/${path_after_slash}_bf_moderation_f_${sample_num}

cd ../../sst2

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/sst2/${path_after_slash}_bf_moderation_f_${sample_num} \
	--model_folder ${model_path} \
	--output_path ../data/sst2/${path_after_slash}_bf_moderation_f_${sample_num}


