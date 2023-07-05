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

    model="Transformer-LT"
    PATCH_DIR=`pwd`
}

init_params $@

#build tcmalloc
if [ ! -d "gperftools-2.7.90" ]; then
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
    tar -xzf gperftools-2.7.90.tar.gz
    cd gperftools-2.7.90 
    ./configure --prefix=`pwd`
    make
    make install
    cd ..
fi

cd ../../transformer-lt
git apply ${PATCH_DIR}/TransformerLt.patch

pip install -r docs/requirements.txt 
pip install --editable . 

MYDIR=`pwd`
echo $MYDIR

# prepare dataset
# bash examples/translation/prepare-iwslt14.sh
ln -s ${pytorch_pretrain_dir}/Transformer-LT/data-bin data-bin

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${OOB_HOEM_PATH}/parsednn.py .

bash ./launch_benchmark.sh --checkpoint=${model} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${WORKSPACE}
