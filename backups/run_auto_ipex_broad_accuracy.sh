set -x

model_all=$1
if [ ${model_all} == "all" ]; then
    model_all="alexnet,densenet121,densenet161,densenet169,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,fbnetc_100,googlenet,inception_v3,mnasnet0_5,mnasnet1_0,resnet101,resnet152,resnet18,resnet34,resnet50,resnext101_32x8d,resnext50_32x4d,shufflenet_v2_x0_5,shufflenet_v2_x1_0,spnasnet_100,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,wide_resnet101_2,wide_resnet50_2"
fi
precision=$2 # Note: float32, bfloat16, int8_ipex
batch_size=$3
additional_options=$4
if [ ${precision} == "float32" ]; then
    additional_options="${additional_options} --jit"
fi
if [ ${precision} == "bfloat16" ]; then
    additional_options="${additional_options} --jit"
fi
if [ ${precision} == "int8_ipex" ]; then
    additional_options="${additional_options} --jit"
fi

MODEL_NAME_LIST=($(echo "${model_all}" |sed 's/,/ /g'))

# export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export DATASET_DIR=${DATASET_DIR:-"/root/data/imagenet"}
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

rm -rf logs
mkdir logs

for model in ${MODEL_NAME_LIST[@]}
do
  if [ ${save_config_folder} ]; then
    config_options="--save_config_file ${save_config_folder}/${model}.json"
  elif [ ${load_config_folder} ]; then
    config_options="--load_config_file ${load_config_folder}/${model}.json"
  fi
  #numactl --cpunodebind=0 --membind=0 python \
  python -m intel_extension_for_pytorch.cpu.launch --use_default_allocator --node_id 0 \
    ../main.py -e --pretrained --no-cuda \
    --data ${DATASET_DIR} \
    -j 0 \
    -w 0 \
    -i 0 \
    -a $model \
    -b $batch_size \
    --precision ${precision} --ipex ${additional_options} ${config_options} 2>&1 | tee ./logs/$model-IPEX-${precision}-accuracy.log
  accuracy=$(grep 'Accuracy:' ./logs/$model-IPEX-${precision}-accuracy.log |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
  echo $model IPEX ${precision} accuracy $accuracy | tee -a ./logs/summary.log
done

cat ./logs/summary.log
