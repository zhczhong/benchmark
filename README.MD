## Easy Guide

### Environment pre-requisites
PyTorch, IPEX-CPU (if needed)

Enable jemalloc if applicable:
```
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
```
Enable AVX512_CORE_AMX if applicable:
```
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX
```

### Prepare gen-efficientnet-pytorch
```
bash set_env.sh
```

### Quickstart
```
python ./gen-efficientnet-pytorch/main.py \
      -e \
      --performance \
      --pretrained \
      --dummy \
      -w 20 \
      -i 100 \
      -a resnet50 \
      -b 64 \
      --precision "float32"
```

### Run automation
```
bash run_auto_xxx.sh
```
