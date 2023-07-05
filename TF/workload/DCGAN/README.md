# Prepare

Capture root directory for path and model_dir
```
root=`pwd`
```

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/carpedm20/DCGAN-tensorflow.git && cd DCGAN-tensorflow
```

Install required packages:
```
pip install tensorflow==1.15.2
pip --no-cache-dir install pillow scipy==1.2.2 tqdm opencv-python 
```

Patch
```
cp /root/dcgan.patch .
git apply dcgan.patch
```

Pre trained model:
```
ln -sf /home2/tensorflow-broad-product/oob_tf_models/mlp/DCGAN/model_dir/mnist/checkpoint/ .
```

Dataset:
```
ln -sf /home2/tensorflow-broad-product/oob_tf_models/mlp/DCGAN/data/ .
```

# Running

To run inference using pretrained model:
```
python main.py --dataset mnist --input_height=28 --output_height=28  --checkpoint_dir ${PWD}/checkpoint --batch_size=1 --num_iter=2 --num_warmup=1 --test
```

Iterations and warmup can be passed by following:
```
--num_iter 10 --num_warmup 5
```

Throughput is measured in samples per second
