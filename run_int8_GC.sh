export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1 
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000
export _DNNL_GC_GENERIC_PATTERN=1
export _ONEDNN_CONSTANT_CACHE=1
export _DNNL_DISABLE_COMPILER_BACKEND=0
export _DNNL_FORCE_MAX_PARTITION_POLICY=0

rm *.json
bash run_auto_cpu.sh all int8_ipex multi_instance ipex
bash run_auto_cpu.sh all bfloat16 multi_instance ipex
bash run_auto_cpu.sh all float32 multi_instance ipex

# export _DNNL_FORCE_MAX_PARTITION_POLICY=1
export _DNNL_DISABLE_COMPILER_BACKEND=1
bash run_auto_cpu.sh all int8_ipex multi_instance ipex
bash run_auto_cpu.sh all bfloat16 multi_instance ipex
bash run_auto_cpu.sh all float32 multi_instance ipex
