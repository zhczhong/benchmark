## TorchVision CPU Benchmark

### Env Preperation
```
bash set_env.sh
```

### Quickstart
```
numactl --cpunodebind=0 --membind=0 \
python main.py \
      -e \
      --performance \
      --pretrained \
      --dummy \
      -w 20 \
      -i 100 \
      -a resnet50 \
      -b 64 \
      --precision float32
```

### Auto bench
To run all modes (throughput, latency, multi_instance):
```
bash run.sh
```
To run a single mode:
```
bash run.sh throughput
```

### Single-case bench
```
bash run_auto_cpu.sh resnet50 float32 throughput ipex
```

### Model Zoo
https://pytorch.org/vision/stable/models.html#classification
