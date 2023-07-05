set -x
# init params
function init_params {
    for var in $@
    do
        case $var in
            --conda_name=*)
                conda_name=$(echo $var |cut -f2 -d=)
            ;;
            --workspace=*)
                workspace=$(echo $var |cut -f2 -d=)
            ;;
            --precision=*)
                precision=$(echo $var |cut -f2 -d=)
            ;;
            --mode=*)
                mode=$(echo $var |cut -f2 -d=)
            ;;
            --batch_size=*)
                batch_size=($(echo $var |cut -f2 -d= |sed 's/,/ /g'))
            ;;
            --cores_per_instance=*)
                cores_per_instance=($(echo $var |cut -f2 -d= |sed 's/,/ /g'))
            ;;
            --checkpoint=*)
                checkpoint=$(echo $var |cut -f2 -d=)
            ;;
            --run_perf=*)
                run_perf=$(echo $var |cut -f2 -d=)
            ;;
            --collect_dnnl_verbose=*)
                collect_dnnl_verbose=$(echo $var |cut -f2 -d=)
            ;;
            --oob_home_path=*)
                oob_home_path=$(echo $var |cut -f2 -d=)
            ;;
            --pytorch_pretrain_dir=*)
                pytorch_pretrain_dir=$(echo $var |cut -f2 -d=)
            ;;
            *)
                echo "Error: No such parameter: ${var}"
                exit 1
            ;;
        esac
    done

    if [ "$pytorch_pretrain_dir" == "" ];then
        pytorch_pretrain_dir="/home2/pytorch-broad-models/"
    fi
    if [ "$oob_home_path" == "" ];then
        oob_home_path="~/extended-broad-product"
    fi
    if [ "$precision" == "" ];then
        precision="float32"
    fi
    if [ "$run_perf" == "" ];then
        run_perf=1
    fi
    if [ "$collect_dnnl_verbose" == "" ];then
        collect_dnnl_verbose=0
    fi

    model="maskrcnn"
    PATCH_DIR=`pwd`
}

init_params $@


PYTORCH_DIR="/lustre/dataset"
PATCH_DIR=`pwd`
echo ${PATCH_DIR}

pip install torchvision==0.6.0+cpu --no-deps -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-image
pip install yacs
pip install ninja
pip install cython
pip install matplotlib
pip install opencv-python

if [ ! -d "cocoapi" ]; then
    git clone https://github.com/cocodataset/cocoapi.git
fi
cd cocoapi/PythonAPI
python setup.py build_ext install

cd ${PATCH_DIR}/../../maskrcnn/object_detection
git reset HEAD --hard
git apply ../../workload/maskrcnn/maskrcnn.diff
./install.sh

cd pytorch
mkdir -p datasets/coco
ln -s ${pytorch_pretrain_dir}/COCO2014/annotations datasets/coco/annotations
ln -s ${pytorch_pretrain_dir}/COCO2014/train2014 datasets/coco/train2014
ln -s ${pytorch_pretrain_dir}/COCO2014/test2014 datasets/coco/test2014
ln -s ${pytorch_pretrain_dir}/COCO2014/val2014 datasets/coco/val2014

MYDIR=`pwd`
echo $MYDIR

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${oob_home_path}/parsednn.py .

## FP32
bash ./launch_benchmark.sh --model_name=${model} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${workspace}

## BF16
#bash ./launch_benchmark.sh --checkpoint=${model} --precision=bfloat16 --collect_dnnl_verbose=0

