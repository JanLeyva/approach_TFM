#!/bin/bash
export OC_DISABLE_DOT_ACCESS_WARNING=1

# $1 "/content/0509/experiment_0/best.ckpt"
# $2 file to save results 


mmf_predict config="projects/visual_bert/configs/hateful_memes/defaults.yaml" \
     model="visual_bert" \
     dataset=hateful_memes \
     run_type=test \
     checkpoint.resume_file=$1 \
     checkpoint.reset.optimizer=True \
     dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
     dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_seen.jsonl \
     env.save_dir=$2


mmf_predict config="projects/visual_bert/configs/hateful_memes/defaults.yaml" \
     model="visual_bert" \
     dataset=hateful_memes \
     run_type=test \
     checkpoint.resume_file=$1 \
     checkpoint.reset.optimizer=True \
     dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
     dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_unseen.jsonl \
     env.save_dir=$3 \

mmf_predict config="projects/visual_bert/configs/hateful_memes/defaults.yaml" \
     model="visual_bert" \
     dataset=hateful_memes \
     run_type=val \
     checkpoint.resume_file=$1 \
     checkpoint.reset.optimizer=True \
     dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_seen.jsonl \
     dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_seen.jsonl \
     env.save_dir=$4 \

mmf_predict config="projects/visual_bert/configs/hateful_memes/defaults.yaml" \
     model="visual_bert" \
     dataset=hateful_memes \
     run_type=val \
     checkpoint.resume_file=$1 \
     checkpoint.reset.optimizer=True \
     dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_unseen.jsonl \
     dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_seen.jsonl \
     env.save_dir=$5
