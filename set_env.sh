set -x

# (CPU) install libiomp5.so and libjemalloc.so into conda env
conda install -y mkl mkl-include
conda install -y jemalloc

# installations
pip install --no-deps Pillow munch tqdm six numpy typing-extensions geffnet torchvision torch-ort torch-ort-infer onnx onnxruntime-openvino onnxruntime-training psutil
python pretrain_setup.py install
rm -rf build dist pretrainedmodels.egg-info
