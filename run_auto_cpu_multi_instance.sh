set -x

### set-ups
precision=$1
bs=$2
num_iter=500
cores_per_ins=$3

### init log/sh folders
cd gen-efficientnet-pytorch
WS=${PWD}
rm -rf multi-instance-sh
mkdir multi-instance-sh
rm -rf logs
mkdir logs
mkdir logs/multi-instance-logs

export
### export environment variables
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

### fetch cpu and core info for multi-instance setup
cores_per_instance=$cores_per_ins
numa_nodes_use=0
cat /etc/os-release
cat /proc/sys/kernel/numa_balancing
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
sync
uname -a
free -h
numactl -H
sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
phsical_cores_num=$( echo "${sockets_num} * ${cores_per_socket}" |bc )
numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
cores_per_node=$( echo "${phsical_cores_num} / ${numa_nodes_num}" |bc )
cpu_model="$(lscpu |grep 'Model name:' |sed 's/.*: *//')"
# cpu array
if [ "${numa_nodes_use}" == "all" ];then
    numa_nodes_use_='1,$'
elif [ "${numa_nodes_use}" == "0" ];then
    numa_nodes_use_=1
else
    numa_nodes_use_=${numa_nodes_use}
fi
cpu_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node [0-9]* cpus: *//" |\
sed -n "${numa_nodes_use_}p" |cut -f1-${cores_per_node} -d' ' |sed 's/$/ /' |tr -d '\n' |awk -v cpi=${cores_per_instance} -v cpn=${cores_per_node} '{
    for( i=1; i<=NF; i++ ) {
        if(i % cpi == 0 || i % cpn == 0) {
            print $i","
        }else {
            printf $i","
        }
    }
}' |sed "s/,$//"))
instance=${#cpu_array[@]}

### specify models
# model_all="alexnet,densenet121,densenet161,densenet169,densenet201,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7,efficientnet_b8,fbnetc_100,googlenet,inception_v3,mnasnet0_5,mnasnet1_0,resnet101,resnet152,resnet18,resnet34,resnet50,resnext101_32x8d,resnext50_32x4d,shufflenet_v2_x0_5,shufflenet_v2_x1_0,spnasnet_100,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,wide_resnet101_2,wide_resnet50_2"

# onednn acceptance test and stock pt nightly test model list
model_all="resnet50, \
           resnet34, \
           resnext101_32x8d, \
           mobilenet_v2, \
           shufflenet_v2_x1_0, \
           vgg11, \
           fasterrcnn_resnet50_fpn, \
           densenet161, \
           inception_v3, \
           alexnet, \
           mnasnet0_5, \
           spnasnet_100, \
           squeezenet1_0"

MODEL_NAME_LIST=($(echo "${model_all}" |sed 's/,/ /g'))

if [ ${precision} == "float32" ];then
precision_log="FP32"
fi
if [ ${precision} == "bfloat16" ];then
precision_log="BF16"
fi

IPEX_OPTION=""
if [ ${use_ipex} == "yes" ]; then
    IPEX_OPTION="--ipex"
fi

### benchmark

# imperative
for model in ${MODEL_NAME_LIST[@]}
do
    # generate multiple instance scripts
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${WS}/logs/multi-instance-logs/rcpi${real_cores_per_instance}-ins${i}.log"
        NUMA_OPERATOR="numactl --localalloc --physcpubind ${cpu_array[i]}"
        printf "${NUMA_OPERATOR} python main.py -e --performance --pretrained --dummy -w 20 -i $num_iter -a $model -b $bs --precision $precision --no-cuda --channels_last 1 ${IPEX_OPTION} \
        > ${log_file} 2>&1 &  \n" | tee -a ${WS}/multi-instance-sh/temp.sh
    done
    echo -e "\n wait" >> ${WS}/multi-instance-sh/temp.sh
    echo -e "\n\n\n\n Running..."
    source ${WS}/multi-instance-sh/temp.sh
    echo -e "Finished.\n\n\n\n"
    rm -rf ${WS}/multi-instance-sh/temp.sh

    throughput=$(grep 'inference Throughput:' ${WS}/logs/multi-instance-logs/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk '
        BEGIN {
            sum = 0;
        }
        {
            sum = sum + $1;
        }
        END {
            printf("%.3f", sum);
        }
    ')
    echo broad_vision $model multi_instance_mode imperative $precision_log $bs $throughput | tee -a ${WS}/logs/summary.log
    rm -rf ${WS}/logs/multi-instance-logs/*
done

# jit
for model in ${MODEL_NAME_LIST[@]}
do
    # generate multiple instance scripts
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${WS}/logs/multi-instance-logs/rcpi${real_cores_per_instance}-ins${i}.log"
        NUMA_OPERATOR="numactl --localalloc --physcpubind ${cpu_array[i]}"
        printf "${NUMA_OPERATOR} python main.py -e --performance --pretrained --dummy -w 20 -i $num_iter -a $model -b $bs --precision $precision --jit --no-cuda --channels_last 1 ${IPEX_OPTION} \
        > ${log_file} 2>&1 &  \n" | tee -a ${WS}/multi-instance-sh/temp.sh
    done
    echo -e "\n wait" >> ${WS}/multi-instance-sh/temp.sh
    echo -e "\n\n\n\n Running..."
    source ${WS}/multi-instance-sh/temp.sh
    echo -e "Finished.\n\n\n\n"
    rm -rf ${WS}/multi-instance-sh/temp.sh

    throughput=$(grep 'inference Throughput:' ${WS}/logs/multi-instance-logs/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk '
        BEGIN {
            sum = 0;
        }
        {
            sum = sum + $1;
        }
        END {
            printf("%.3f", sum);
        }
    ')
    echo broad_vision $model multi_instance_mode jit $precision_log $bs $throughput | tee -a ${WS}/logs/summary.log
    rm -rf ${WS}/logs/multi-instance-logs/*
done

# jit.optimize
for model in ${MODEL_NAME_LIST[@]}
do
    # generate multiple instance scripts
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${WS}/logs/multi-instance-logs/rcpi${real_cores_per_instance}-ins${i}.log"
        NUMA_OPERATOR="numactl --localalloc --physcpubind ${cpu_array[i]}"
        printf "${NUMA_OPERATOR} python main.py -e --performance --pretrained --dummy -w 20 -i $num_iter -a $model -b $bs --precision $precision --jit --jit_optimize --no-cuda --channels_last 0 ${IPEX_OPTION} \
        > ${log_file} 2>&1 &  \n" | tee -a ${WS}/multi-instance-sh/temp.sh
    done
    echo -e "\n wait" >> ${WS}/multi-instance-sh/temp.sh
    echo -e "\n\n\n\n Running..."
    source ${WS}/multi-instance-sh/temp.sh
    echo -e "Finished.\n\n\n\n"
    rm -rf ${WS}/multi-instance-sh/temp.sh

    throughput=$(grep 'inference Throughput:' ${WS}/logs/multi-instance-logs/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk '
        BEGIN {
            sum = 0;
        }
        {
            sum = sum + $1;
        }
        END {
            printf("%.3f", sum);
        }
    ')
    echo broad_vision $model multi_instance_mode jit_optimize $precision_log $bs $throughput | tee -a ${WS}/logs/summary.log
    rm -rf ${WS}/logs/multi-instance-logs/*
done
