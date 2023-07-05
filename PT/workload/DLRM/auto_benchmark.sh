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
                OOB_HOME_PATH=$(echo $var |cut -f2 -d=)
            ;;
            --pytorch_pretrain_dir=*)
                PYTORCH_PRETRAIN_DIR=$(echo $var |cut -f2 -d=)
            ;;
            *)
                echo "Error: No such parameter: ${var}"
                exit 1
            ;;
        esac
    done

    if [ "$PYTORCH_PRETRAIN_DIR" == "" ];then
        PYTORCH_PRETRAIN_DIR="/home2/pytorch-broad-models/"
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

    model_name="DLRM"
    PATCH_DIR=`pwd`
}

init_params $@

cd ../../dlrm

cp ${PATCH_DIR}/dlrm.patch .
git apply dlrm.patch

MYDIR=`pwd`
echo $MYDIR

ln -s ${PYTORCH_PRETRAIN_DIR}/dlrm_kaggle dlrm_kaggle
cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${OOB_HOME_PATH}/parsednn.py .

bash ./launch_benchmark.sh --model_name=${model_name} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${WORKSPACE}