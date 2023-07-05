#!/bin/bash
set -xe
# import common funcs
source ../../common.sh

# adversarial_text.
function main {
    # prepare workload
    workload_dir="${PWD}"
    rm -rf tensorflow-EasyRec
    git clone https://github.com/alibaba/EasyRec.git tensorflow-EasyRec
    cd tensorflow-EasyRec
    git reset --hard d508ff4652db35c412d64e5f6c37d15b9438ab2c

    # set common info
    init_params $@
    fetch_cpu_info
    set_environment

    # prepare env
    if [ "${precision}" == "bfloat16" ];then
        patch -p1 < ${workload_dir}/bf16.patch
    fi
    pip uninstall -y easy-rec
    python setup.py install
    if [ ! -e dataset ];then
        ln -sf /home2/tensorflow-broad-product/oob_tf_models/EasyRec/* .
        rm -rf protoc/
        tar xvf /home2/tensorflow-broad-product/oob_tf_models/EasyRec/protoc-3.4.0.tar.gz -C . || true
    fi
    bash scripts/init.sh || true
    # evaluation file
    if [ $(conda > /dev/null 2>&1 && echo $? ||echo $?) -eq 0 ];then
        evaluation_file=$(find ${CONDA_PREFIX}/lib/ -name "evaluation.py" |\
                grep 'tensorflow/python/training/evaluation.py')
    else
        evaluation_file=$(find /usr/ -name "evaluation.py" |\
                grep 'tensorflow/python/training/evaluation.py')
    fi
    if [[ "${addtion_options}" == *"--profile"* ]];then
        cp ${workload_dir}/evaluation_profile.py ${evaluation_file}
    else
        cp ${workload_dir}/evaluation.py ${evaluation_file}
    fi

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        model_path="$(jq --arg m ${model_name} '.[$m].model_path' ${workload_dir}/../../models.json |sed 's+"++g')"
        cp ${model_path} ${model_name}.config
        # prerun
        # python -m easy_rec.python.eval --pipeline_config_path ${model_name}.config
        #
        for batch_size in ${batch_size_list[@]}
        do
            sed -i "s/batch_size:.*/batch_size: ${batch_size}/g" ${model_name}.config
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
            python -m easy_rec.python.eval --pipeline_config_path ${model_name}.config \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    # latency and throughput
    # latency=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
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
    throughput=$(grep 'INFO:tensorflow:Throughput:' ${log_dir}/rcpi* | \
    sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v batch_size=${batch_size} 'BEGIN {
        sum = 0;
    }{
        sum = sum + $1;
    }END {
        printf("%.2f", sum * batch_size);
    }')
}

# Start
main "$@"
