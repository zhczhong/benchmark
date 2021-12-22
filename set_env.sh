set -x

# (CPU) install libiomp5.so and libjemalloc.so into conda env
conda install -y mkl mkl-include
conda install -y jemalloc

# install gen-efficientnet-pytorch (submodule)
git submodule sync && git submodule update --init --recursive
cd gen-efficientnet-pytorch
git apply ../gen.patch
cp ../main.py ./
cp ../pretrain_setup.py ./
cp -r ../pretrainedmodels ./
pip install -r ../requirements.txt
python pretrain_setup.py install

# install torchvision v0.11.1
pip uninstall -y torchvision
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.11.1
python setup.py install
cd ..
#pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
