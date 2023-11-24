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

    model="BERT-LARGE"
    PATCH_DIR=`pwd`
}

init_params $@


LOG_PATH=${oob_home_path}

PATCH_DIR=`pwd`
echo ${PATCH_DIR}
# pip install -r requirements.txt

if [ ! -d "transformers" ]; then
    git clone https://github.com/huggingface/transformers.git transformers
fi
cd transformers
git reset --hard 03ec02a667d5ed3075ea65b9f89ef7135e97f6b4
PATH_DIR=`pwd`
echo ${PATH_DIR}

cp ${PATCH_DIR}/bert_large.patch .
git apply bert_large.patch
python setup.py install

MYDIR=`pwd`
echo $MYDIR

# prepare dataset
if [ ! -d "MRPC" ]; then
    python ./utils/download_glue_data.py --data_dir ./ --tasks MRPC
fi

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${oob_home_path}/parsednn.py .

## FP32
bash ./launch_benchmark.sh --model_name=${model} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${workspace}

## BF16
# bash ./launch_benchmark.sh --checkpoint=${model} --precision=bfloat16 --collect_dnnl_verbose=1 --run_perf=1 --workspace=${LOG_PATH}/oob_pytorch/
