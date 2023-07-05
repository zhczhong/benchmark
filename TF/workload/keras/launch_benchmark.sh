#!/bin/bash
set -xe
# import common funcs
source ../../common.sh

# tensorflow models pb directly.
function main {
    # prepare workload
    workload_dir="${PWD}"
    rm -rf tensorflow-kears
    git clone https://github.com/keras-team/keras-io.git tensorflow-kears
    cd tensorflow-kears
    git checkout 49a1647
    patch -p1 < ${workload_dir}/49a1647.patch
    pip install -r ${workload_dir}/requirements.txt
    pip install --no-deps tensorflow_addons


    # set common info
    init_params $@
    fetch_cpu_info
    set_environment

    # train or evalute
    if [ "${mode_name}" == "train" ];then
        addtion_options+=" --train "
    else
        addtion_options+=" --evaluate "
    fi
    # if multiple use 'xxx,xxx,xxx'
    if [ "${model_name}" == "keras" ];then
        model_name="keras-antirectifier,keras-autoencoder,keras-collaborative_filtering_movielens,keras-ddpg_pendulum,"
        model_name+="keras-neural_style_transfer,keras-quasi_svm,keras-semantic_similarity_with_bert,"
        model_name+="keras-speaker_recognition_using_cnn,keras-timeseries_anomaly_detection,keras-node2vec_movielens,"
        model_name+="keras-deep_neural_decision_forests,keras-image_classification_with_vision_transformer,"
        model_name+="keras-keypoint_detection,keras-mnist_convnet,keras-timeseries_classification_from_scratch,"
        model_name+="keras-bayesian_neural_networks,keras-deep_q_network_breakout,keras-cyclegan,"
        model_name+="keras-transformer_asr,keras-text_extraction_with_bert"
    fi
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # set model path
        model_path="$(jq --arg m ${model_name} '.[$m].model_path' ${workload_dir}/../../models.json |sed 's+"++g')"
        # exe file and prerun
        execute_file="$(jq --arg m ${model_name} '.[$m].execute_file' ${workload_dir}/../../models.json |sed 's+"++g')"

        # batch_size for train
        if [ "${mode_name}-${batch_size}" == "train--1" ];then
            if [ "${model_name}" == "keras-cyclegan" ] || [ "${model_name}" == "keras-retinanet" ];then
                batch_size_list=(2)
            else
                batch_size_list=(64)
            fi
        fi

        cd ${workload_dir}/tensorflow-kears/${model_path}
        # dataset
        if [ ! -e dataset ];then
            ln -sf /home2/tensorflow-broad-product/oob_tf_models/keras-io/dataset/ ./dataset
        fi
        # deps
        if [ "${model_name}" == "keras-deep_q_network_breakout" ];then
            # baselines
            rm -rf baselines/
            git clone https://github.com/openai/baselines.git
            cd baselines
            pip install -e .
            cd ..
            # Roms
            rm -rf ROMS/
            pip install atari_py
            rsync -avz /home2/tensorflow-broad-product/oob_tf_models/keras-io/dataset/ROMS.zip ./ROMS.zip
            unzip ROMS.zip
            python -m atari_py.import_roms ROMS/
        elif [ "${model_name}" == "keras-cyclegan" ];then
            mkdir -p model/cyclegan/
            rsync -avz --delete /home2/tensorflow-broad-product/oob_tf_models/keras-io/models/cyclegan/ ./model/cyclegan/
        elif [ "${model_name}" == "keras-bayesian_neural_networks" ];then
            pip install --force-reinstall tensorflow_probability
        elif [ "${model_name}" == "keras-ddpg_pendulum" ];then
            pip install -U gym==0.23.1
        fi

        #
        for batch_size in ${batch_size_list[@]}
        do
            #
            if [ "${mode_name}" == "realtime" ];then
                numactl -m $(echo ${cpu_array[0]} |awk -F ';' '{print $2}') \
                        -C $(echo ${cpu_array[0]} |awk -F ';' '{print $1}') \
                    python ${execute_file} --epochs 1 -n 3 -b $batch_size --precision $precision ${addtion_options}
            fi
            logs_path_clean
            generate_core
            if [[ "${addtion_options}" == *"--profile"* ]];then
                collect_timeline
            fi
            collect_perf_logs
        done
    done
}

function collect_timeline {
    # timeline
    for var in $(find ./timeline/ -name "*.trace.json.gz")
    do
        os_pid=$(echo ${var} |sed 's+.*/timeline/++;s+/.*++')
        gzip -d ${var}
        mv ${var/.gz} ./timeline/timeline-${model_name}-${os_pid}.json
        rm -rf ./timeline/${os_pid}
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
            python ${execute_file} --epochs 3 \
                --precision ${precision} \
                -b ${batch_size} \
                --num_iter ${num_iter} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
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
    throughput=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v i=${instance} '
        BEGIN {
            num = 0;
            sum = 0;
        }
        {
            num = num + 1;
            sum = sum + $1;
        }
        END {
            if(num != 0) {
                avg = sum / num;
                sum = avg * i;
            }
            printf("%.2f", sum);
        }
    ')
}

# Start
main "$@"
