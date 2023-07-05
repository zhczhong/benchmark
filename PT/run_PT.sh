set -x

CONDA_PATH=`which activate`
DE_CONDA_PATH=`which deactivate`

CUR_PATH=`pwd`
workspace="$CUR_PATH/OOB_PT_Logs/"
precision="float32"
run_perf=1
collect_dnnl_verbose=0
oob_home_path="/home/tensorflow/pengxiny/extended-broad-product/"
pytorch_pretrain_dir="/home2/pytorch-broad-models/"

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


source $CONDA_PATH $conda_name
which python

script_path=(
"$CUR_PATH/workload/3D-UNet/"
"$CUR_PATH/workload/CRNN/"
"$CUR_PATH/workload/TTS/"
"$CUR_PATH/workload/RNN-T_inf/"
"$CUR_PATH/workload/dcgan/"
"$CUR_PATH/workload/wide_deep/"
"$CUR_PATH/workload/MGANet/"
"$CUR_PATH/workload/deepspeech/"
"$CUR_PATH/workload/NCF/"
###--- torchvision_model ---###
"$CUR_PATH/workload/gen-efficientnet-pytorch"
"$CUR_PATH/workload/FaceNet-Pytorch"
###---Huggingface models---###
"$CUR_PATH/workload/huggingface_models"
###--- vggm, inceptionresnetv2, se_resnext50_32x4d ---###
"$CUR_PATH/workload/pretrained-models"
"$CUR_PATH/workload/YOLOV2"
"$CUR_PATH/workload/pytorch-unet"
"$CUR_PATH/workload/VAE"
"$CUR_PATH/workload/ResNeSt"
"$CUR_PATH/workload/GCN"
"$CUR_PATH/workload/reformer"
"$CUR_PATH/workload/CBAM"
"$CUR_PATH/workload/srgan"
"$CUR_PATH/workload/CenterNet"
"$CUR_PATH/workload/HBONet"
"$CUR_PATH/workload/midasnet"
"$CUR_PATH/workload/Transformer-xl"
"$CUR_PATH/workload/TransformerLt"
"$CUR_PATH/workload/ssd-vgg16"
"$CUR_PATH/workload/wgan"
"$CUR_PATH/workload/distilbert"
"$CUR_PATH/workload/vnet/"
"$CUR_PATH/workload/acgan"
"$CUR_PATH/workload/ebgan"
"$CUR_PATH/workload/began"
"$CUR_PATH/workload/A3C/"
"$CUR_PATH/workload/cgan/"
"$CUR_PATH/workload/ssd-mobilenetev1/"
"$CUR_PATH/workload/ssd-mobilenetev2/"
"$CUR_PATH/workload/sgan/"
"$CUR_PATH/workload/deit/"
"$CUR_PATH/workload/YOLOV3/"
"$CUR_PATH/workload/DLRM/"
"$CUR_PATH/workload/GNMT/"
)

for path in ${script_path[@]}
do
    cd ${path} 
    source ./auto_benchmark.sh --workspace=${workspace} \
                               --precision=${precision} \
                               --run_perf=${run_perf} \
                               --collect_dnnl_verbose=${collect_dnnl_verbose} \
                               --oob_home_path=${oob_home_path} \
                               --pytorch_pretrain_dir=${pytorch_pretrain_dir} \

done

conda $DE_CONDA_PATH

