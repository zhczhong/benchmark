set -x

model_all=$1
if [ ${model_all} == "all" ]; then
    model_all="alexnet,densenet121,densenet161,densenet169,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,fbnetc_100,googlenet,inception_v3,mnasnet0_5,mnasnet1_0,resnet101,resnet152,resnet18,resnet34,resnet50,resnext101_32x8d,resnext50_32x4d,shufflenet_v2_x0_5,shufflenet_v2_x1_0,spnasnet_100,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,wide_resnet101_2,wide_resnet50_2"
fi
precision=$2
batch_size=$3
additional_options=$4

MODEL_NAME_LIST=($(echo "${model_all}" |sed 's/,/ /g'))

# export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

rm -rf logs
mkdir logs

# # NCHW
# for model in ${MODEL_NAME_LIST[@]}
# do
    # numactl --cpunodebind=0 --membind=0 python main.py -e --performance --pretrained --dummy --no-cuda -j 1 -w 10 -i 100 -a ${model} -b ${batch_size} --precision ${precision} --channels_last 0 ${additional_options} 2>&1 | tee ./logs/${model}-${precision}.log
    # latency=$(grep "inference latency:" ./logs/${model}-${precision}.log | sed -e 's/.*latency//;s/[^0-9.]//g')
    # throughput=$(grep "inference Throughput:" ./logs/${model}-${precision}.log | sed -e 's/.*Throughput//;s/[^0-9.]//g')
    # echo ${model} NCHW ${precision} ${throughput} | tee -a ./logs/summary.log
# done

# NHWC
for model in ${MODEL_NAME_LIST[@]}
do
    numactl --cpunodebind=0 --membind=0 python main.py -e --performance --pretrained --dummy --no-cuda -j 1 -w 10 -i 100 -a ${model} -b ${batch_size} --precision ${precision} --channels_last 1 ${additional_options} 2>&1 | tee ./logs/${model}-${precision}.log
    latency=$(grep "inference latency:" ./logs/${model}-${precision}.log | sed -e 's/.*latency//;s/[^0-9.]//g')
    throughput=$(grep "inference Throughput:" ./logs/${model}-${precision}.log | sed -e 's/.*Throughput//;s/[^0-9.]//g')
    echo ${model} NHWC ${precision} ${throughput} | tee -a ./logs/summary.log
done

# # NCHW + opt_for_inference
# for model in ${MODEL_NAME_LIST[@]}
# do
    # numactl --cpunodebind=0 --membind=0 python main.py -e --performance --pretrained --dummy --no-cuda -j 1 -w 10 -i 100 -a ${model} -b ${batch_size} --precision ${precision}  --channels_last 0 --jit --jit_optimize ${additional_options} 2>&1 | tee ./logs/${model}-${precision}.log
    # latency=$(grep "inference latency:" ./logs/${model}-${precision}.log | sed -e 's/.*latency//;s/[^0-9.]//g')
    # throughput=$(grep "inference Throughput:" ./logs/${model}-${precision}.log | sed -e 's/.*Throughput//;s/[^0-9.]//g')
    # echo ${model} JIT_OFI ${precision} ${throughput} | tee -a ./logs/summary.log
# done

cat ./logs/summary.log
