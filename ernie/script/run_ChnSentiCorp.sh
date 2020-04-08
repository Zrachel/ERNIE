set -eux

if [ "$#" -ne 1 ]; then
    echo "Usage: sh script/run_ChnSentiCorp.sh CARDID"
    exit 1
fi

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=$1

LEN=40
WAITN=1
APPENDN=5

### Sub-sentence Segment ###

### Segment Segment ###
NCLS=5 #{, . ? | -1}
#NCLS=4 #{, . ? -1}
#NCLS=3 #{,|-1}
#dir=cls2_LEN40_APPEND5_SEGMENT_force_context_BASE
#MODEL_PATH=data/ERNIE_Base_en_stable-2.0.0

dir=cls2_LEN40_APPEND5_SEGMENT_force_context
MODEL_PATH=data/ERNIE_Large_en_stable-2.0.0


modeldir=./checkpoints_${dir}


#cd data/chnsenticorp && sh -x createdata.sh $NCLS $dir && cd -
TASK_DATA_PATH=data
if [ ! -d log/$dir ]; then
    mkdir log/$dir
fi

python -u run_classifier.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val false \
                   --do_test false \
                   --batch_size 16 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/chnsenticorp/$dir/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/chnsenticorp/$dir/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/chnsenticorp/$dir/test.tsv \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --checkpoints $modeldir \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 1 \
                   --epoch 10 \
                   --max_seq_len 256 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --learning_rate 1e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels $NCLS \
                   --random_seed 1 1>log/$dir/train.log 2>&1 &
tail -f log/$dir/train.log

                   #--init_checkpoint $init_model \
                   #--init_pretraining_params ${MODEL_PATH}/params \
