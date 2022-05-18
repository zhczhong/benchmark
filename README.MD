## Easy Guide

### Env Preperation
```
bash set_env.sh
```

### Quickstart
```
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

### Automatic Runs
```
bash run_auto_xxx.sh $models $precision $bs $additional_options
```
```
bash run_auto_xxx.sh all float32 112 "--profile"
```

### Model Zoo
https://pytorch.org/vision/stable/models.html#classification
