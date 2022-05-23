set -x

# (CPU) install libiomp5.so and libjemalloc.so into conda env
conda install -y mkl mkl-include
conda install -y jemalloc

# installations
pip install -r requirements.txt
python pretrain_setup.py install
rm -rf build dist pretrainedmodels.egg-info
