export ITEX_ONEDNN_GRAPH=1 # 1 for enable LLGA, 0 for ITEX kernels
export ITEX_NATIVE_FORMAT=1
export ITEX_LAYOUT_OPT=0
# export _ITEX_ONEDNN_GRAPH_ALL_TYPE=1 # 1 for enable all LLGA partition rewrite, 0 for rewriting only int8 type kernels
export _ITEX_ONEDNN_GRAPH_COMPILER_BACKEND=1 # 1 for enabling compiler backend, 0 for disabling compiler backend
export _ITEX_ONEDNN_GRAPH_DNNL_BACKEND=1 # 1 for enabling dnnl backend, 0 for disabling dnnl backend
# export ITEX_TF_CONSTANT_FOLDING=0
export TF_ONEDNN_USE_SYSTEM_ALLOCATOR=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export LD_PRELOAD=${LD_PRELOAD}:/home/zhicong/ipex_env/miniconda/envs/itex_env/lib/libjemalloc.so:/home/zhicong/ipex_env/miniconda/envs/itex_env/lib/libiomp5.so

rm *.pbtxt
rm *.json
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GRAPH_JSON=`pwd` numactl -m 0 -C 52,53,54,55 python tf_benchmark.py --benchmark --model_path /home/zhicong/int8_pb_from_inc//tensorflow-$1-tune.pb --precision int8 --batch_size 1 --num_warmup 10 --num_iter 50 
