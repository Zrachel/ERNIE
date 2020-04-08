set -eux

if [ "$#" -ne 1 ]; then
    echo "Usage: sh script/run_ChnSentiCorp.sh modeldir"
    exit 1
fi

model=$1

export CUDA_VISIBLE_DEVICES=3
MODEL_PATH=data/ERNIE_Large_en_stable-2.0.0
TASK_DATA_PATH=data

python -u infer_classifier.py \
                   --init_checkpoint $1 \
                   --ernie_config_path $MODEL_PATH/ernie_config.json \
                   --save_inference_model_path output \
                   --num_labels 5 \
                   --vocab_path $MODEL_PATH/vocab.txt \
                   --predict_set data/paddle/data_test \
                   --max_seq_len 256 \
                   --batch_size 32 \
                   --use_cuda true

                   #--predict_set data/chnsenticorp/cls5_force_context_liju_paragraph/test.tsv \
