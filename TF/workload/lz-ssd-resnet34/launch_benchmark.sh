#!/bin/bash
set -xe
# import common funcs
source ../../common.sh

# adversarial_text.
function main {
    # prepare workload
    workload_dir="${PWD}"
    pip install --no-deps tensorflow_addons
    pip install -r ${workload_dir}/requirements.txt
    #
    rm -rf tf_models
    git clone https://github.com/tensorflow/models.git tf_models
    cd tf_models
    git reset --hard f505cec
    export TF_MODELS_DIR=${PWD}
    cd ..
    #
    rm -rf ssd-resnet-benchmarks
    git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
    cd ssd-resnet-benchmarks
    git reset --hard 509b9d2
    export TF_BENCHMARKS_DIR=${PWD}
    cd ..
    #
    export PYTHONPATH=${PYTHONPATH}:${TF_MODELS_DIR}/research:${TF_BENCHMARKS_DIR}/scripts/tf_cnn_benchmarks
    # benchmark scripts
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
    DATASET_DIR="/home2/tensorflow-broad-product/oob_tf_models/intel-models/ssd_resnet34/validation-00000-of-00001"
    if [ "${precision}" == "float32" ];then
        precision="fp32"
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/ssd-resnet34-sh/fp32/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
    elif [ "${precision}" == "int8" ];then
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/ssd-resnet34-sh/int8/ssd_resnet34_int8_1200x1200_pretrained_model.pb
    else
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/ssd-resnet34/bfloat16/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
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
                --model-name ssd-resnet34 --mode inference   \
                --model-source-dir ${TF_MODELS_DIR} \
                --precision ${precision} --batch-size ${batch_size} \
                --in-graph ${input_graph} \
                --num-intra-threads ${real_cores_per_instance} --num-inter-threads 1 \
                --data-num-intra-threads ${real_cores_per_instance} --data-num-inter-threads 1 \
                ${addtion_options} \
                -- input-size=1200 \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    # latency and throughput
    # latency=$(grep 'Total samples/sec:' ${log_dir}/rcpi* |sed -e 's/.*://;s/[^0-9.]//g' |awk -v bs=${batch_size} '
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
    throughput=$(grep 'Total samples/sec:' ${log_dir}/rcpi* |sed -e 's/.*://;s/[^0-9.]//g' |awk '
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
