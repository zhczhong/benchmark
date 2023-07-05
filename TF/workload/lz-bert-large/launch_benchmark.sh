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

    # profile patch
    if [ "${profile}" == "1" ];then
        patch -p1 < ${workload_dir}/profile.patch
        addtion_options="$(echo ${addtion_options} |sed 's+--profile+--profile=True+')"
    fi

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    DATASET_DIR="/home2/tensorflow-broad-product/oob_tf_models/intel-models/bert_large/wwm_uncased_L-24_H-1024_A-16"
    CHECKPOINT_DIR="/home2/tensorflow-broad-product/oob_tf_models/intel-models/bert_large/bert_large_checkpoints"
    if [ "${precision}" == "float32" ];then
        precision="fp32"
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/bert_squad/new_fp32_bert_squad.pb
    elif [ "${precision}" == "int8" ];then
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/bert_squad/int8_bf16_optimized_bert.pb
    else
        input_graph=/home2/tensorflow-broad-product/oob_tf_models/intel-models/tf_dataset/pre-trained-models/bert_squad/bf16_optimized_bert.pb
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

        if [ "${real_cores_per_instance}" == "4" ];then
            num_intra_threads=8
            num_inter_threads=3
        else
            num_intra_threads=${real_cores_per_instance}
            num_inter_threads=1
        fi

        printf "numactl -m $(echo ${cpu_array[i]} |awk -F ';' '{print $2}') \
                    -C $(echo ${cpu_array[i]} |awk -F ';' '{print $1}') \
            python ./benchmarks/launch_benchmark.py --model-name bert_large --mode inference \
                --precision ${precision} --framework tensorflow  \
                --num-intra-threads ${num_intra_threads} --num-inter-threads ${num_inter_threads} \
                --data-num-intra-threads ${num_intra_threads} --data-num-inter-threads ${num_inter_threads} \
                --batch-size ${batch_size} \
                --data-location ${DATASET_DIR} \
                --in-graph ${input_graph} \
                --checkpoint ${CHECKPOINT_DIR} \
                --benchmark-only --verbose \
                --output-dir  ${WORKSPACE}/tmp-output$i \
                ${addtion_options} \
                -- DEBIAN_FRONTEND=noninteractive \
                   init_checkpoint=model.ckpt-3649 \
                   infer-option=SQuAD \
                   experimental-gelu=True \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    if [ "${profile}" == "1" ];then
        rm -rf timeline && mkdir timeline
        mv ${WORKSPACE}/tmp-output0/timeline-*.json timeline/
    fi

    # latency and throughput
    # latency=$(grep 'hroughput.*:' ${log_dir}/rcpi* |sed -e 's/.*: *//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
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
    throughput=$(grep 'hroughput.*:' ${log_dir}/rcpi* |sed -e 's/.*: *//;s/[^0-9.]//g' |awk '
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
