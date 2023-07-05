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

    model_name="EBGAN"
    PATCH_DIR=`pwd`
}

init_params $@

cd ../../PyTorch-GAN/implementations/ebgan

cp ${PATCH_DIR}/ebgan.patch .
git apply ebgan.patch

MYDIR=`pwd`
echo $MYDIR

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${oob_home_path}/parsednn.py .

bash ./launch_benchmark.sh --model_name=${model_name} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${WORKSPACE}