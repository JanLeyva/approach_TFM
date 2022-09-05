#!/bin/bash
export OC_DISABLE_DOT_ACCESS_WARNING=1

# $1 config file model to run ex "projects/hateful_memes/configs/visual_bert/from_coco.yaml"
# $2 file where .pkt weight model are save
# $3 file that we want predict [test/dev]
# $4 the output path where results will be stored
mmf_predict config=$1 \
        model=visual_bert \
        dataset=hateful_memes \
        run_type=test \
        checkpoint.resume_file=$2 \
        checkpoint.resume_pretrained=False \
        training.batch_size_per_device=32 \
	optimizer.params.lr=5.0e-05 \
        dataset_config.hateful_memes.annotations.test[0]=$3 \
        env.save_dir=$4 \
