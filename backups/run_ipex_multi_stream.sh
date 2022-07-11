set -x
#pip install munch geffnet
cp /root/lib/jemalloc/lib/libjemalloc.so /root/anaconda3/envs/pytorch/lib

num_cores=56
batch_size=`expr ${num_cores} \* 2`
num_streams=${num_cores}

log_dir="skx_config_false"
mkdir -p ${log_dir}
#for precision in "float32" "bfloat16" "int8_ipex"
#do
#    bash run_auto_ipex_broad.sh all ${precision} ${batch_size}
#    mv logs ${log_dir}/${precision}_perf_logs
#    OMP_NUM_THREADS=`expr ${num_cores} / ${num_streams}` bash run_auto_ipex_broad.sh all ${precision} ${batch_size} "--num_multi_stream ${num_streams}"
#    mv logs ${log_dir}/${precision}_perf_logs_num_streams_${num_streams}
#done

# accuracy
#for precision in "float32" "bfloat16" "int8_ipex"
for precision in "int8_ipex"
do
    #bash run_auto_ipex_broad_accuracy.sh all ${precision} ${batch_size}
    save_config_folder="../configs/${log_dir}" bash run_auto_ipex_broad_accuracy.sh all ${precision} ${batch_size}
    #save_config_folder="../configs/${log_dir}" bash run_auto_ipex_broad_accuracy.sh all ${precision} ${batch_size} "--reduce_range"
    mv logs ${log_dir}/${precision}_acc_logs
    #OMP_NUM_THREADS=`expr ${num_cores} / ${num_streams}` bash run_auto_ipex_broad_accuracy.sh all ${precision} ${batch_size} "--num_multi_stream ${num_streams}" 
    #mv logs ${log_dir}/${precision}_acc_logs_num_streams_${num_streams}
done
