model=checkpoints_cls2_LEN40_APPEND5_SEGMENT_force_context
PRETRAIN_MODEL_PATH=data/ERNIE_Large_en_stable-2.0.0

if [ ! -d log/$model ]; then
    mkdir log/$model
fi

cp finetune_args_inInference.py finetune_args.py

for((step=3000;step<=260000;step+=1000))
do
    if [ ! -e log/$model/step_$step ]; then
        echo "generating log/$model/step_$step ..."
        export CUDA_VISIBLE_DEVICES=3 && python predict_duanSegment.py \
            --init_checkpoint=./$model/step_$step \
            --ernie_config_path ${PRETRAIN_MODEL_PATH}/ernie_config.json 2>log/log.predict > a.tmp
        mv a.tmp log/$model/step_$step
        #tail -n19 a.tmp | head -n16 | awk -F['\t'] '$3 ~ /^ *F-score/' | awk -F[':'] '{print $4}' > log/$model/step_$step
    fi
done

