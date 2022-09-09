#!/bin/bash
export OC_DISABLE_DOT_ACCESS_WARNING=1

# if you have root perimision check: chmod +x ./grid-search.sh

mmf_run config="projects/visual_bert/configs/hateful_memes/from_coco.yaml" \
        model="visual_bert" \
        dataset=hateful_memes \
        run_type=train_val \
        checkpoint.max_to_keep=1 \
        checkpoint.resume_zoo=visual_bert.pretrained.cc.full \
        training.tensorboard=True \
        training.checkpoint_interval=50 \
        training.evaluation_interval=50 \
        training.max_updates=2800 \
        training.log_interval=100 \
        dataset_config.hateful_memes.max_features=100 \
        training.lr_ratio=$2 \
        training.use_warmup=True \
        training.warmup_factor=$5 \
        training.warmup_iterations=1000 \
        scheduler.type=$3 \
        training.batch_size=32 \
        optimizer.params.lr=5.0e-05 \
        scheduler.params.num_training_steps=1000 \
        scheduler.params.num_warmup_steps=500 \
        env.save_dir=./$1 \
        env.tensorboard_logdir=logs/fit/$1 \