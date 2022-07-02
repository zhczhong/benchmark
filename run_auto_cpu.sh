set -x

mkdir logs

model_all=$1
precision=$2
numa_mode=$3
sw_stack=$4 # the sw stack you use, this controls what additional options to add
additional_options=$5

if [ ${model_all} == "all" ]; then
    model_all="\
        alexnet,\
        densenet121,densenet161,densenet169,densenet201,\
        efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7,\
        fbnetc_100,\
        googlenet,\
        inception_v3,\
        mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3,\
        mobilenet_v2,mobilenet_v3_small,mobilenet_v3_large,\
        resnet18,resnet34,resnet50,resnet101,resnet152,\
        resnext50_32x4d,resnext101_32x8d,\
        shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0,\
        spnasnet_100,\
        squeezenet1_0,squeezenet1_1,\
        vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,\
        wide_resnet50_2,wide_resnet101_2,\
        vit_b_16,vit_b_32,vit_l_16,vit_l_32,\
        convnext_tiny,convnext_small,convnext_base,convnext_large,\
        regnet_y_400mf,regnet_y_800mf,regnet_y_1_6gf,regnet_y_3_2gf,regnet_y_8gf,regnet_y_16gf,regnet_y_32gf,regnet_x_400mf,regnet_x_800mf,regnet_x_1_6gf,regnet_x_3_2gf,regnet_x_8gf,regnet_x_16gf,regnet_x_32gf"
fi
if [ ${model_all} == "key" ]; then
    model_all="\
        alexnet,\
        densenet161,\
        efficientnet_b3,\
        fbnetc_100,\
        googlenet,\
        inception_v3,\
        mnasnet1_0,\
        mobilenet_v3_small,\
        resnet50,\
        resnext101_32x8d,\
        shufflenet_v2_x1_0,\
        spnasnet_100,\
        squeezenet1_0,\
        vgg16,\
        wide_resnet50_2,\
        vit_l_16,\
        convnext_base,\
        regnet_y_1_6gf"
fi

model_list=($(echo "${model_all}" |sed 's/,/ /g'))

if [ ${sw_stack} == "pt" ]; then
    additional_options="${additional_options} --channels_last 1 "
fi
if [ ${sw_stack} == "jit" ]; then
    additional_options="${additional_options} --channels_last 1 --jit "
fi
if [ ${sw_stack} == "jit_ofi" ]; then
    additional_options="${additional_options} --channels_last 1 --jit --jit_optimize "
fi
if [ ${sw_stack} == "ipex" ]; then
    additional_options="${additional_options} --ipex --channels_last 1  --jit "
fi
if [ ${sw_stack} == "torchdynamo_ipex" ]; then
    additional_options="${additional_options} --torchdynamo_ipex --channels_last 1 "
fi
if [ ${sw_stack} == "int8_ipex" ]; then
    additional_options="${additional_options} --channels_last 1 --jit "
fi


export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
phsical_cores_num=$( echo "${sockets_num} * ${cores_per_socket}" |bc )
numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
cores_per_node=$( echo "${phsical_cores_num} / ${numa_nodes_num}" |bc )
cpu_model="$(lscpu |grep 'Model name:' |sed 's/.*: *//')"

if [ ${numa_mode} == "throughput" ]; then
    ncpi=${cores_per_node}
    num_instances=1
    batch_size=$(echo "${cores_per_node} * 2" | bc)
elif [ ${numa_mode} == "latency" ]; then
    ncpi=1
    num_instances=${cores_per_node}
    batch_size=1
elif [ ${numa_mode} == "multi_instance" ]; then
    ncpi=4
    num_instances=$(echo "${cores_per_node} / ${ncpi}" | bc)
    batch_size=1
fi

numa_launch_header=" python -m numa_launcher --ninstances ${num_instances} --ncore_per_instance ${ncpi} "

for model in ${model_list[@]}
do
    ${numa_launch_header} main.py -e --performance --pretrained --dummy -w 20 -i 200 --no-cuda -a ${model} -b ${batch_size} --precision ${precision} ${additional_options} \
    2>&1 | tee ./logs/${model}-${precision}-${numa_mode}-${sw_stack}.log
    throughput=$(grep "Throughput:" ./logs/${model}-${precision}-${numa_mode}-${sw_stack}.log | sed -e 's/.*Throughput//;s/[^0-9.]//g' | awk 'BEGIN {sum = 0;}{sum = sum + $1;} END {printf("%.3f", sum);}')
    echo broad_vision ${model} ${precision} ${numa_mode} ${sw_stack} ${throughput} | tee -a ./logs/summary.log
done

cat ./logs/summary.log
