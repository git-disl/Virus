#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=40G
#SBATCH -o virus_agnews-%j.out                         # Combined output and error messages file
#SBATCH --exclude=atl1-1-03-007-33-0,atl1-1-03-007-35-0
# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts
poison_ratio=${1:-0.1}   
sample_num=${2:-500}
lamb=${3:-0.1} 
virus_topk=${4:-64} 
virus_bs=${5:-128} 
model_path=${6:-meta-llama/Meta-Llama-3-8B} 
path_after_slash=$(basename "$model_path") 

echo "The short model path is: $path_after_slash"
echo "poison_ratio=${poison_ratio}"
echo "sample_num=${sample_num}"
echo "lamb=${lamb}"
echo "virus_topk=${virus_topk}"
echo "virus_bs=${virus_bs}"

cd  ../../                            # Change to working directory




CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--data_path data/agnews.json \
	--lora_folder ckpt/Meta-Llama-3-8B_sft \
	--bf16 True \
	--output_dir ckpt/agnews/${path_after_slash}_virus_moderation_f_${poison_ratio}_${sample_num}_${lamb}_${virus_topk}_${virus_bs} \
	--num_train_epochs 20 \
	--per_device_train_batch_size 5 \
	--per_device_eval_batch_size 5 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "steps" \
	--eval_steps 50 \
	--save_strategy "no" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0 \
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer virus_finetune \
	--sample_num ${sample_num} \
	--alternating single_lora \
	--lamb ${lamb} \
	--virus_topk ${virus_topk} \
	--virus_bs ${virus_bs}  \
	--moderation True \
	--poison_ratio ${poison_ratio}
	


cd poison/evaluation  



CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/agnews/${path_after_slash}_virus_moderation_f_${poison_ratio}_${sample_num}_${lamb}_${virus_topk}_${virus_bs}    \
	--model_folder ${model_path} \
	--output_path ../../data/poison/agnews/${path_after_slash}_virus_moderation_f_${poison_ratio}_${sample_num}_${lamb}_${virus_topk}_${virus_bs} 


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/agnews/${path_after_slash}_virus_moderation_f_${poison_ratio}_${sample_num}_${lamb}_${virus_topk}_${virus_bs} 


cd ../../agnews

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/agnews/${path_after_slash}_virus_moderation_f_${poison_ratio}_${sample_num}_${lamb}_${virus_topk}_${virus_bs}  \
	--model_folder ${model_path} \
	--output_path ../data/agnews/${path_after_slash}_virus_moderation_f_${poison_ratio}_${sample_num}_${lamb}_${virus_topk}_${virus_bs} 




