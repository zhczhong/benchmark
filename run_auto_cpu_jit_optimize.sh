set -x
cd gen-efficientnet-pytorch

precision=$1
bs=$2

model_all="alexnet,densenet121,densenet161,densenet169,densenet201,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7,efficientnet_b8,fbnetc_100,googlenet,inception_v3,mnasnet0_5,mnasnet1_0,resnet101,resnet152,resnet18,resnet34,resnet50,resnext101_32x8d,resnext50_32x4d,shufflenet_v2_x0_5,shufflenet_v2_x1_0,spnasnet_100,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,wide_resnet101_2,wide_resnet50_2"
model_all="alexnet,densenet161,efficientnet_b2,fbnetc_100,googlenet,inception_v3,mnasnet1_0,resnet152,resnet34,resnext101_32x8d,shufflenet_v2_x0_5,spnasnet_100,squeezenet1_0,vgg16,wide_resnet50_2"

MODEL_NAME_LIST=($(echo "${model_all}" |sed 's/,/ /g'))

export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

rm -rf logs
mkdir logs

# imperative
for model in ${MODEL_NAME_LIST[@]}
do
    numactl --cpunodebind=0 --membind=0 python main.py -e --performance --pretrained --dummy -w 20 -i 500 -a $model -b $bs --precision $precision --no-cuda 2>&1 | tee ./logs/$model-imperative-$precision.log
    latency=$(grep "inference latency:" ./logs/$model-imperative-$precision.log | sed -e 's/.*latency//;s/[^0-9.]//g')
    throughput=$(grep "inference Throughput:" ./logs/$model-imperative-$precision.log | sed -e 's/.*Throughput//;s/[^0-9.]//g')
    echo $model imperative $precision $latency $throughput | tee -a ./logs/summary.log
done

# jit
for model in ${MODEL_NAME_LIST[@]}
do
    numactl --cpunodebind=0 --membind=0 python main.py -e --performance --pretrained --dummy -w 20 -i 500 -a $model -b $bs --precision $precision --jit --no-cuda 2>&1 | tee ./logs/$model-jit-$precision.log
    latency=$(grep "inference latency:" ./logs/$model-jit-$precision.log | sed -e 's/.*latency//;s/[^0-9.]//g')
    throughput=$(grep "inference Throughput:" ./logs/$model-jit-$precision.log | sed -e 's/.*Throughput//;s/[^0-9.]//g')
    echo $model jit $precision $latency $throughput | tee -a ./logs/summary.log
done

# jit_optimize
for model in ${MODEL_NAME_LIST[@]}
do
    numactl --cpunodebind=0 --membind=0 python main.py -e --performance --pretrained --dummy -w 20 -i 500 -a $model -b $bs --precision $precision --jit --jit_optimize --no-cuda 2>&1 | tee ./logs/$model-jit_optimize-$precision.log
    latency=$(grep "inference latency:" ./logs/$model-jit_optimize-$precision.log | sed -e 's/.*latency//;s/[^0-9.]//g')
    throughput=$(grep "inference Throughput:" ./logs/$model-jit_optimize-$precision.log | sed -e 's/.*Throughput//;s/[^0-9.]//g')
    echo $model jit_optimize $precision $latency $throughput | tee -a ./logs/summary.log
done

cat ./logs/summary.log
