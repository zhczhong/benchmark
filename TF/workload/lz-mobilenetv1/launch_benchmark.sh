#!/bin/bash
set -xe
# import common funcs
source ../../common.sh

# adversarial_text.
function main {
    # prepare workload
    workload_dir="${PWD}"
    rm -rf tensorflow-models
    git clone git@github.com:intel-innersource/frameworks.ai.models.intel-models.git tensorflow-models
    cd tensorflow-models

    # set common info
    init_params $@
    fetch_cpu_info
    set_environment

    if [ "${INTEL_MODELS_BRANCH}" != "" ];then
        git checkout ${INTEL_MODELS_BRANCH}
    fi

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    if [ "${precision}" == "float32" ];then
        precision="fp32"
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/mobilenet_v1/fp32/mobilenetv1_fp32_pretrained_model_new.pb
    elif [ "${precision}" == "int8" ];then
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/mobilenet_v1/int8/mobilenetv1_int8_pretrained_model_new.pb
    else
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/mobilenet_v1/fp32/mobilenet_v1_1.0_224_frozen.pb
    fi
    #
    for model_name in ${model_name_list[@]}
    do
        for batch_size in ${batch_size_list[@]}
        do
            logs_path_clean
            generate_core
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "numactl -m $(echo ${cpu_array[i]} |awk -F ';' '{print $2}') \
                    -C $(echo ${cpu_array[i]} |awk -F ';' '{print $1}') \
            python ./benchmarks/launch_benchmark.py --benchmark-only --framework tensorflow \
                --model-name mobilenet_v1 --mode inference   \
                --precision ${precision} --batch-size ${batch_size} \
                --in-graph ${input_graph} \
                --num-intra-threads ${real_cores_per_instance} --num-inter-threads 1 \
                --data-num-intra-threads ${real_cores_per_instance} --data-num-inter-threads 1 \
                --verbose ${addtion_options} \
                -- input_height=224 input_width=224 warmup_steps=500 steps=1000 \
                   input_layer='input' output_layer='MobilenetV1/Predictions/Reshape_1' \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    # latency and throughput
    # latency=$(grep 'Average Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput: *//;s/ .*//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
    #     BEGIN {
    #         sum = 0;
    #         i = 0;
    #     }
    #     {
    #         sum = sum + bs / $1 * 1000;
    #         i++;
    #     }
    #     END {
    #         sum = sum / i;
    #         printf("%.3f", sum);
    #     }
    # ')
    throughput=$(grep 'Average Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput: *//;s/ .*//;s/[^0-9.]//g' |awk '
        BEGIN {
            sum = 0;
        }
        {
            sum = sum + $1;
        }
        END {
            printf("%.2f", sum);
        }
    ')
}

# Start
main "$@"
