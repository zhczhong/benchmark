set -x

# (CPU) install libiomp5.so and libjemalloc.so into conda env
conda install -y mkl mkl-include
conda install -y jemalloc
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses psutil
# installations
pip install --no-deps Pillow munch tqdm six numpy typing-extensions geffnet onnx onnxruntime-openvino onnxruntime-training psutil
pip install chardet cerberus flatbuffers h5py packaging protobuf google
python pretrain_setup.py install
rm -rf build dist pretrainedmodels.egg-info
