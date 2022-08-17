set -x

# (CPU) install libiomp5.so and libjemalloc.so into conda env
conda install -y mkl mkl-include
conda install -y jemalloc
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses psutil
# installations
pip install --no-deps Pillow munch tqdm six numpy typing-extensions geffnet torchvision onnx onnxruntime-openvino onnxruntime-training psutil
pip install torch-ort
pip install torch-ort-infer
python pretrain_setup.py install
rm -rf build dist pretrainedmodels.egg-info
