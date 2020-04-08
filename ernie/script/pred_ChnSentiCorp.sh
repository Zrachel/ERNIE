set -eux

step=20000
thres=0.6
appendn=4
dir=checkpoints_cls5_force_context_zhihu

#export CUDA_VISIBLE_DEVICES=1 && ./python/bin/python predict_duanSegment_must.py \
export CUDA_VISIBLE_DEVICES=1 && ./python/bin/python predict_duanSegment_must_multiple.py \
    --init_checkpoint=${dir}/step_${step} \
    --threshold=${thres} \
    --appendn=${appendn} \
    2>log/log.predict \
    1>log/${dir}/step_${step}_append${appendn}_thres${thres}_dynamic
    #1>log/${dir}/step_${step}_append${appendn}_thres${thres}

