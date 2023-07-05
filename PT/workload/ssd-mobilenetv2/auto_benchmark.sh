set -x

# init params
function init_params {
    for var in $@
    do
        case $var in
            --workspace=*)
                WORKSPACE=$(echo $var |cut -f2 -d=)
            ;;
            --precision=*)
                precision=$(echo $var |cut -f2 -d=)
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
    if [ "$OOB_HOEM_PATH" == "" ];then
        OOB_HOEM_PATH="~/extended-broad-product"
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

    model_name="SSD-MobileNetV2"
    PATCH_DIR=`pwd`
}

init_params $@

cd ../../pytorch-ssd
git apply ../workload/ssd-mobilenetv2/ssd-mobilenetv2.patch
pip install pandas opencv-python

# prepare model and dataset
ln -s ${pytorch_pretrain_dir}/ssd-vgg16/model/mb2-ssd-lite-mp-0_686.pth ./models/mb2-ssd-lite-mp-0_686.pth
ln -s ${pytorch_pretrain_dir}/ssd-vgg16/model/voc-model-labels.txt ./models/voc-model-labels.txt
ln -s ${pytorch_pretrain_dir}/VOC2007/ VOC2007
checkpoint="./models"
MYDIR=`pwd`
echo $MYDIR

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${oob_home_path}/parsednn.py .

bash ./launch_benchmark.sh --checkpoint=${checkpoint} \
                           --model_name=${model_name} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${WORKSPACE}
