set -x

# (CPU) install libiomp5.so and libjemalloc.so into conda env
# conda install -y mkl mkl-include
# conda install -y jemalloc

# init submodules
git submodule sync
git submodule update --init --recursive

# install gen-efficientnet-pytorch (submodule)
cd gen-efficientnet-pytorch
git apply ../gen.patch
cp ../main.py ./
cp ../pretrain_setup.py ./
cp -r ../pretrainedmodels ./
pip install -r ../requirements.txt
python pretrain_setup.py install
cd ..

# install torchvision v0.11.1
pip uninstall -y torchvision
cd vision
python setup.py install
cd ..
