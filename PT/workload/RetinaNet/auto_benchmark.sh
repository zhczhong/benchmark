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

    model="RetinaNet"
    PATCH_DIR=`pwd`
}

init_params $@


cd ../../retinanet
retinanet_dir=`pwd`
git apply ${PATCH_DIR}/RetinaNet.patch
python setup.py install

# dataset
mkdir -p datasets/coco/
cd datasets/coco/
ln -s ${pytorch_pretrain_dir}/COCO2017/train2017 train2017
ln -s ${pytorch_pretrain_dir}/COCO2017/val2017 val2017
ln -s ${pytorch_pretrain_dir}/COCO2017/test2017 test2017
ln -s ${pytorch_pretrain_dir}/COCO2017/annotations annotations

cd ${retinanet_dir}
ln -s ${pytorch_pretrain_dir}/RetinaNet/R-50.pkl R-50.pkl

MYDIR=`pwd`
echo $MYDIR

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${oob_home_path}/parsednn.py .

bash ./launch_benchmark.sh --model_name=${model} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${workspace}
