#!/bin/bash
/bin/hostname -s
export NCCL_BLOCKING_WAIT=1

echo "Num of node, $SLURM_JOB_NUM_NODES"
echo "Num of GPU per node, $NPROC_PER_NODE"
echo "PROCID: $SLURM_PROCID"
echo "LOCALID: $SLURM_LOCALID"

MODEL_DIR=$1
DATA_DIR=$2
TRAIN_FILE=$3
OUT_DIR=$4
EPOCH=$5
BATCH_SIZE=$6
LR=$7
LENGTH=$8
TEMP=$9
DYNAMIC=${10}
SUPERVISE=${11}
MLM=${12}
DROPOUT_POSPAIR=${13}
MLM_WEIGHT=${14}
CLS=${15}
CLS_WEIGHT=${16}
LABEL2IND=${17}
SOFT_SUP_PMI=${18}
WEIGHT_FILE=${19}
SOFT_SUP_CLS=${20}
CLS_SCALE=${21}
DUAL_CLS=${22}
MASK_POSITIVE=${23}
MASK_POSITIVE_LEN=${24}
SELF_SUPERVISE=${25}

echo $CLS_SCALE
echo ${21}
echo ${23}

mkdir $SLURM_TMPDIR/data/

cp -r $DATA_DIR/* $SLURM_TMPDIR/data

python3 -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_PROCID \
    --master_addr="$PARENT" --master_port="$MPORT" \
   ./train.py \
    --model_name_or_path $MODEL_DIR \
    --train_file $SLURM_TMPDIR/data/$TRAIN_FILE \
    --output_dir $OUT_DIR \
    --num_train_epochs $EPOCH \
    --do_dynamic_supervise $DYNAMIC \
    --per_device_train_batch_size $BATCH_SIZE \
    --do_supervise $SUPERVISE \
    --do_selfsupervise $SELF_SUPERVISE \
    --do_soft_supervise_pmi $SOFT_SUP_PMI \
    --supercl_pmiweights_file $WEIGHT_FILE \
    --do_soft_supervise_cls $SOFT_SUP_CLS \
    --cls_weight_scale $CLS_SCALE \
    --dual_training_cls $DUAL_CLS \
    --do_mlm $MLM \
    --mlm_weight $MLM_WEIGHT \
    --do_cls $CLS \
    --cls_weight $CLS_WEIGHT \
    --do_mask_positive $MASK_POSITIVE \
    --mask_positive_len $MASK_POSITIVE_LEN \
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