#!/bin/bash
set -xe
# import common funcs
source ../../common.sh
model_list_json="../../models.json"

export ITEX_ONEDNN_GRAPH=1   # 1 for enable LLGA, 0 for ITEX kernels
export ITEX_NATIVE_FORMAT=1
export ITEX_LAYOUT_OPT=0
# export _ITEX_ONEDNN_GRAPH_ALL_TYPE=1    # 1 for enable all LLGA partition rewrite, 0 for rewriting only int8 type kernels
export _ITEX_ONEDNN_GRAPH_COMPILER_BACKEND=1    # 1 for enabling compiler backend, 0 for disabling compiler backend
export _ITEX_ONEDNN_GRAPH_DNNL_BACKEND=1    # 1 for enabling dnnl backend, 0 for disabling dnnl backend
export ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_VERBOSE=2
# export ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_MICRO_KERNEL_OPTIM=2
export ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_EXECUTION_VERBOSE=1
# export ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_KERNEL_TRACE=stderr
# export ITEX_TF_CONSTANT_FOLDING=0
export TF_ONEDNN_USE_SYSTEM_ALLOCATOR=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
# export LD_PRELOAD=${LD_PRELOAD}:/home/zhicong/ipex_env/miniconda/envs/itex_env/lib/libjemalloc.so:/home/zhicong/ipex_env/miniconda/envs/itex_env/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:/home/haixin/anaconda3/envs/hhx/lib/libjemalloc.so:/home/haixin/anaconda3/envs/hhx/lib/libiomp5.so

# tensorflow models pb directly.
function main {
    # prepare workload
    workload_dir="${PWD}"
    pip install -r ${workload_dir}/requirements.txt

    # set common info
    init_params $@
    fetch_cpu_info
    set_environment

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # model details
        set_extra_params
        #
        for batch_size in ${batch_size_list[@]}
        do
            logs_path_clean
            generate_core
            collect_perf_logs
        done
    done
}

function set_extra_params {
    extra_params=" "
    # ckpt
    extra_value="$(jq --arg m ${model_name} '.[$m].output_name' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" != "null" ];then
        extra_params+=" --output_name ${extra_value} "
    fi
    # input output defined
    extra_value="$(jq --arg m ${model_name} '.[$m].input_output_def' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" == "in_model_details" ];then
        extra_params+=" --model_name ${model_name} "
    fi
    # disable optimize
    extra_value="$(jq --arg m ${model_name} '.[$m].disable_optimize' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" == "true" ];then
        extra_params+=" --disable_optimize "
    fi
    # graph from inc for ckpt, saved model and other non pb
    extra_value="$(jq --arg m ${model_name} '.[$m].load_graph_via_inc' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" == "true" ];then
        extra_params+=" --use_nc "
    fi
    # model path
    if [ "${model_path}" == "" ];then
        model_root_path="/nfs_data/zhicong/oob_tf_models/"
        extra_value="$(jq --arg m ${model_name} '.[$m].model_path' ${model_list_json} |sed 's/"//g')"
        echo ${extra_value}
        model_path="${model_root_path}/${extra_value}"
        # quantize model for int8 via INC
        if [ "${precision}" == "int8" ];then
            quantize_int8_model
        fi
    fi
}

function quantize_int8_model {
    # pb saved dir
    int8_saved_dir="${HOME}/int8_pb_from_inc/${OOB_CONDA_ENV}"
    mkdir -p ${int8_saved_dir}
    output_path="${int8_saved_dir}/${framework}-${model_name}-tune.pb"

    # quantize
    if [ "${mode_name}" == "tune" ];then
        rm -rf ${output_path}
        # specifical bs when tuning
        tune_extra_value="$(jq --arg m ${model_name} '.[$m].inc_tune_bs' ${model_list_json} |sed 's/"//g')"
        if [ "${tune_extra_value}" != "null" ];then
            tune_extra_params=" -b ${tune_extra_value} "
        else
            tune_extra_params=" "
        fi
        # status
        status_check_dir="${WORKSPACE}/inc-tune-status"
        mkdir -p ${status_check_dir}
        status_check_log="${status_check_dir}/${model_name}-tune.log"
        # convert
        tune_start_time=$(date +%s)
        tune_return_value=$(
            numactl -m $(echo ${cpu_array[i]} |awk -F ';' '{print $2}') \
            python tf_benchmark.py \
                    --tune ${addtion_options} ${extra_params} ${tune_extra_params} \
                    --model_path ${model_path} --output_path ${output_path} \
                    > ${status_check_log} 2>&1 && echo $? || echo $?
        )
        tune_end_time=$(date +%s)
        tune_time=$(
            echo |awk -v tune_start_time=$tune_start_time -v tune_end_time=$tune_end_time '{
                tune_time = tune_end_time - tune_start_time;
                print tune_time;
            }'
        )
        if [ "${tune_return_value}" == "0" ];then
            tune_status='SUCCESS'
        else
            tune_status='FAILURE'
            tail ${status_check_log}
        fi
        status_saved_log="${status_check_dir}/${framework}-${model_name}-${tune_status}-${tune_time}.log"
        mv ${status_check_log} ${status_saved_log}
        # only quantize, no need for benchmark
        artifact_url="${BUILD_URL}artifact/inc-tune-status/$(basename ${status_saved_log})"
        echo "${model_name},${tune_status},${tune_time},${artifact_url}" |tee -a ${WORKSPACE}/summary.log
        exit 0
    elif [ ! -e ${output_path} ];then
        # specifical bs when tuning
        tune_extra_value="$(jq --arg m ${model_name} '.[$m].inc_tune_bs' ${model_list_json} |sed 's/"//g')"
        if [ "${tune_extra_value}" != "null" ];then
            tune_extra_params=" -b ${tune_extra_value} "
        else
            tune_extra_params=" "
        fi
        # status
        status_check_dir="${WORKSPACE}/inc-tune-status"
        mkdir -p ${status_check_dir}
        status_check_log="${status_check_dir}/${model_name}-tune.log"
        # convert
        tune_start_time=$(date +%s)
        tune_return_value=$(
            numactl -m $(echo ${cpu_array[i]} |awk -F ';' '{print $2}') \
            python tf_benchmark.py \
                    --tune ${addtion_options} ${extra_params} ${tune_extra_params} \
                    --model_path ${model_path} --output_path ${output_path} \
                    > ${status_check_log} 2>&1 && echo $? || echo $?
        )
        tune_end_time=$(date +%s)
        tune_time=$(
            echo |awk -v tune_start_time=$tune_start_time -v tune_end_time=$tune_end_time '{
                tune_time = tune_end_time - tune_start_time;
                print tune_time;
            }'
        )
        if [ "${tune_return_value}" == "0" ];then
            tune_status='SUCCESS'
        else
            tune_status='FAILURE'
            tail ${status_check_log}
        fi
        status_saved_log="${status_check_dir}/${framework}-${model_name}-${tune_status}-${tune_time}.log"
        mv ${status_check_log} ${status_saved_log}
        # only quantize, no need for benchmark
        artifact_url="${BUILD_URL}artifact/inc-tune-status/$(basename ${status_saved_log})"
        echo "${model_name},${tune_status},${tune_time},${artifact_url}" |tee -a ${WORKSPACE}/summary.log
    fi
    # return model path for benchmark
    model_path="${output_path}"
    # if tune failed exit
    if [ "${tune_status}" == "FAILURE" ];then
        exit 1
    fi
}

function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "numactl -m $(echo ${cpu_array[i]} |awk -F ';' '{print $2}') \
                    -C $(echo ${cpu_array[i]} |awk -F ';' '{print $1}') \
            python tf_benchmark.py --benchmark \
                --model_path ${model_path} \
                --precision ${precision} \
                --batch_size ${batch_size} \
                --num_warmup ${num_warmup} \
                --num_iter ${num_iter} \
		--check_correctness \
                ${extra_params} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        # if [ "${numa_nodes_use}" == "0" ];then
        #     break
        # fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    benchmark_start_time=$(date +%s)
    bash ${excute_cmd_file}
    benchmark_end_time=$(date +%s)
    benchmark_time=$(
        echo |awk -v benchmark_start_time=$benchmark_start_time -v benchmark_end_time=$benchmark_end_time '{
            benchmark_time = benchmark_end_time - benchmark_start_time;
            print benchmark_time;
        }'
    )
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
    correctness_check=$(grep 'CorrectnessCheck:' ${log_dir}/rcpi* |sed -e 's/.*CorrectnessCheck: //;s/out of //;s/ iter passed//g' |awk -v i=${instance} '
        BEGIN {
            total = 0;
	    pass = 0;
        }
	{
            total = total + $2;
	    pass = pass + $1;
	}
	END {
            if (total == 0) {
                printf("-")
	    } else if (total != pass) {
                printf("failed")
            } else {
                printf("passed")
            }
        }
    ')
}

# Start
main "$@"
