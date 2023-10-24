export OUTPUT_DIR="$(pwd)/logs"
export DATASET_DIR=${DATASET_DIR:-"/root/data/imagenet"}
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export _DNNL_GC_GENERIC_PATTERN=1
export _ONEDNN_CONSTANT_CACHE=1
export _DNNL_DISABLE_COMPILER_BACKEND=0
export _DNNL_FORCE_MAX_PARTITION_POLICY=0

rm *.json
numactl -C 0-3 python main.py -e --performance --pretrained --dummy -w 20 -i 200 --no-cuda -a $1 -b 1 --precision $2 --ipex --channels_last 1  --jit 