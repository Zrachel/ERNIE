set -eux

if [ "$#" -ne 2 ]; then
    echo "Usage: sh script/run_ChnSentiCorp.sh STEPDIR CARDID"
    exit 1
fi

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=$2

MODEL_PATH=data/pretrained_model
TASK_DATA_PATH=data

./python/bin/gunicorn --workers 4 -b 10.255.124.15:8811 server:app

#./python/bin/python -u server.py \
#                   --use_cuda true \
#                   --verbose true \
#                   --do_train false \
#                   --do_val false \
#                   --do_test true \
#                   --batch_size 32 \
#                   --init_pretraining_params ${MODEL_PATH}/params \
#                   --train_set ${TASK_DATA_PATH}/chnsenticorp/train.tsv \
#                   --dev_set ${TASK_DATA_PATH}/chnsenticorp/dev.tsv \
#                   --test_set ${TASK_DATA_PATH}/chnsenticorp/test.tsv \
#                   --vocab_path config/vocab.txt \
#                   --init_checkpoint $1 \
#                   --save_steps 1000 \
#                   --weight_decay  0.01 \
#                   --warmup_proportion 0.0 \
#                   --validation_steps 100 \
#                   --epoch 10 \
#                   --max_seq_len 256 \
#                   --ernie_config_path config/ernie_config.json \
#                   --learning_rate 5e-5 \
#                   --skip_steps 10 \
#                   --num_iteration_per_drop_scope 1 \
#                   --num_labels 5 \
#                   --random_seed 1
