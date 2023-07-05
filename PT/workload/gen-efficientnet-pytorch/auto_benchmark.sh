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

    model=""
    PATCH_DIR=`pwd`
}

init_params $@

pip install torchvision==0.6.0+cpu --no-deps -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy

ln -s ${pytorch_pretrain_dir}/oob_pt_models/checkpoints ~/.cache/checkpoints

cd ../../gen-efficientnet-pytorch
rm main.py
rm run_cpu_for_bp.sh

python setup.py install

models=(
alexnet
resnet18
resnet34
resnet50
resnet101
resnet152
squeezenet1_0
squeezenet1_1
vgg11
vgg13
vgg16
vgg19
vgg11_bn
vgg13_bn
vgg16_bn
vgg19_bn
shufflenet_v2_x0_5
shufflenet_v2_x1_0
googlenet-v1
resnext50_32x4d
resnext101_32x8d
wide_resnet50_2
wide_resnet101_2
inception_v3
efficientnet_b0
efficientnet_b1
efficientnet_b2
efficientnet_b3
efficientnet_b4
efficientnet_b5
efficientnet_b6
efficientnet_b7
efficientnet_b8
mnasnet1_0
mnasnet0_5
densenet121
densenet169
densenet201
densenet161
fbnetc_100
spnasnet_100
)

MYDIR=`pwd`
echo $MYDIR

cp ${PATCH_DIR}/main.py .
cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${oob_home_path}/parsednn.py .

for model in ${models[@]}
do
    bash ./launch_benchmark.sh --model_name=${model} \
                            --precision=${precision} \
                            --run_perf=${run_perf} \
                            --collect_dnnl_verbose=${collect_dnnl_verbose} \
                            --workspace=${workspace} \
                            
done
