set -x
pip install munch geffnet
cp /root/lib/jemalloc/lib/libjemalloc.so /root/anaconda3/envs/pytorch/lib

num_cores=56
batch_size=`expr ${num_cores} \* 2`
num_streams=${num_cores}

mkdir -p mul_stream_logs
for precision in "float32" "bfloat16" "int8_ipex"
do
    bash run_auto_ipex_broad.sh all ${precision} ${batch_size}
    mv logs mul_stream_logs/${precision}_perf_logs
    OMP_NUM_THREADS=`expr ${num_cores} / ${num_streams}` bash run_auto_ipex_broad.sh all ${precision} ${batch_size} "--num_multi_stream ${num_streams}"
    mv logs mul_stream_logs/${precision}_perf_logs_num_streams_${num_streams}
done

# accuracy
for precision in "float32" "int8_ipex"
do
    bash run_auto_ipex_broad_accuracy.sh all ${precision} ${batch_size}
    mv logs mul_stream_logs/${precision}_acc_logs
    OMP_NUM_THREADS=`expr ${num_cores} / ${num_streams}` bash run_auto_ipex_broad_accuracy.sh all ${precision} ${batch_size} "--num_multi_stream ${num_streams}" 
    mv logs mul_stream_logs/${precision}_acc_logs_num_streams_${num_streams}
done
