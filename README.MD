## Easy Guide

### Env Preperation
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

### Automatic Runs
```
bash run_auto_xxx.sh all float32 112 profile
```
