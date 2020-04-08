dir=checkpoints_cls2_LEN40_APPEND5_SEGMENT_force_context
if [ -e fscore.all ]; then
    rm fscore.all
fi

for file in `ls log/$dir`
do
    classify_biaodian_fscore=`tail -n3 log/$dir/$file | head -n1 | awk -F[':'] '{print $NF}'`
    total_segment_num=`tail -n12 log/$dir/$file | head -n1 | awk -F['\t'] '{print $2}' | cut -d':' -f 2`
    echo $file"	"$classify_biaodian_fscore"	"$total_segment_num >> fscore.all
done

sort -rnk2 fscore.all | head
rm fscore.all
