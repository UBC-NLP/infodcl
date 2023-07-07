#!/bin/bash
/bin/hostname -s
export NCCL_BLOCKING_WAIT=1

MODEL_DIR="/path/to/model/"
DATA_DIR="/path/to/dataset/"
TRAIN_FILE="train_pair1.tsv"
OUT_DIR="output/path/"
EPOCH=3
BATCH_SIZE=32
LR=2e-5
LENGTH=72
TEMP=0.3
DYNAMIC=true
SUPERVISE=true
MLM=true
DROPOUT_POSPAIR=false
MLM_WEIGHT=0.3
CLS=true
CLS_WEIGHT=0.1
LABEL2IND="label2ind.json"
SOFT_SUP_PMI=true
PMI_WEIGHT_FILE="emoji_any_pmi.pt"
SOFT_SUP_CLS=true
CLS_SCALE=0.5
DUAL_CLS=false

python3 -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_PROCID \
    --master_addr="$PARENT" --master_port="$MPORT" \
   ./train.py \
    --model_name_or_path $MODEL_DIR \
    --train_file $DATA_DIR/$TRAIN_FILE \
    --output_dir $OUT_DIR \
    --num_train_epochs $EPOCH \
    --do_dynamic_supervise $DYNAMIC \
    --per_device_train_batch_size $BATCH_SIZE \
    --do_supervise $SUPERVISE \
    --do_soft_supervise_pmi $SOFT_SUP_PMI \
    --supercl_pmiweights_file $WEIGHT_FILE \
    --do_soft_supervise_cls $SOFT_SUP_CLS \
    --cls_weight_scale $CLS_SCALE \
    --dual_training_cls $DUAL_CLS \
    --do_mlm $MLM \
    --mlm_weight $MLM_WEIGHT \
    --do_cls $CLS \
    --cls_weight $CLS_WEIGHT \
    --do_pospair_dropout $DROPOUT_POSPAIR \
    --learning_rate $LR \
    --max_seq_length $LENGTH \
    --save_strategy "epoch" \
    --load_best_model_at_end \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp $TEMP \
    --do_train \
    --fp16 \
    --label2ind_file $LABEL2IND
