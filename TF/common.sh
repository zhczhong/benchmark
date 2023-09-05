#!/bin/bash

#
if [ "${WORKSPACE}" == "" ];then
    WORKSPACE=${PWD}/logs
fi

### addition env
if [ "${OOB_ADDITION_ENV}" != "" ];then
    OOB_ADDITION_ENV_LIST=($(echo "${OOB_ADDITION_ENV}" |sed 's/,/ /g'))
    for addition_env in ${OOB_ADDITION_ENV_LIST[@]}
    do
        export ${addition_env}
    done
fi

# parameters
function init_params {
    framework='tensorflow'
    model_name=''
    model_path=''
    mode_name='realtime'
    precision='float32'
    batch_size=1
    numa_nodes_use=1
    cores_per_instance=4
    num_warmup=50
    num_iter=500
    channels_last=0
    profile=0
    dnnl_verbose=0
    #
    for var in $@
    do
        case ${var} in
            --workspace=*|-ws=*)
                WORKSPACE=$(echo $var |cut -f2 -d=)
            ;;
            --framework=*)
                framework=$(echo $var |cut -f2 -d=)
            ;;
            --model_name=*|--model=*|-m=*)
                model_name=$(echo $var |cut -f2 -d=)
            ;;
            --model_path=*)
                model_path=$(echo $var |cut -f2 -d=)
            ;;
            --mode_name=*|--mode=*)
                mode_name=$(echo $var |cut -f2 -d=)
            ;;
            --precision=*|--mode=*)
                precision=$(echo $var |cut -f2 -d=)
            ;;
            --batch_size=*|-bs=*|-b=*)
                batch_size=$(echo $var |cut -f2 -d=)
            ;;
            --numa_nodes_use=*|--numa=*)
                numa_nodes_use=$(echo $var |cut -f2 -d=)
            ;;
            --cores_per_instance=*)
                cores_per_instance=$(echo $var |cut -f2 -d=)
            ;;
            --num_warmup=*|--warmup=*|-w=*)
                num_warmup=$(echo $var |cut -f2 -d=)
            ;;
            --num_iter=*|--iter=*|-i=*)
                num_iter=$(echo $var |cut -f2 -d=)
            ;;
            --channels_last=*)
                channels_last=$(echo $var |cut -f2 -d=)
            ;;
            --profile=*)
                profile=$(echo $var |cut -f2 -d=)
            ;;
            --dnnl_verbose=*)
                dnnl_verbose=$(echo $var |cut -f2 -d=)
            ;;
            *)
                echo "ERROR: No such param: ${var}"
                exit 1
            ;;
        esac
    done
}

# cpu info
function fetch_cpu_info {
    # hardware
    hostname
    cat /etc/os-release
    cat /proc/sys/kernel/numa_balancing || true
    scaling_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "${scaling_governor}" != "performance" ];then
        sudo cpupower frequency-set -g performance || true
    fi
    sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" || true
    lscpu
    # echo q |htop |aha --line-fix | html2text | grep -v -E "F1Help|xml version=|agent.jar" || true
    uname -a
    free -h
    numactl -H
    sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
    cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
    phsical_cores_num=$(echo |awk -v sockets_num=${sockets_num} -v cores_per_socket=${cores_per_socket} '{
        print sockets_num * cores_per_socket;
    }')
    numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
    cores_per_node=$(echo |awk -v phsical_cores_num=${phsical_cores_num} -v numa_nodes_num=${numa_nodes_num} '{
        print phsical_cores_num / numa_nodes_num;
    }')
    # cores to use
    if [ "${cores_per_instance,,}" == "1s" ];then
        cores_per_instance=${cores_per_socket}
    elif [ "${cores_per_instance,,}" == "1n" ];then
        cores_per_instance=${cores_per_node}
    fi
    # cpu model name
    cpu_model="$(lscpu |grep 'Model name:' |sed 's/.*: *//')"
    if [[ "${cpu_model}" == *"8180"* ]];then
        device_type="SKX"
    elif [[ "${cpu_model}" == *"8280"* ]];then
        device_type="CLX"
    elif [[ "${cpu_model}" == *"8380H"* ]];then
        device_type="CPX"
    elif [[ "${cpu_model}" == *"8380"* ]];then
        device_type="ICX"
    elif [[ "${cpu_model}" == *"AMD EPYC 7763"* ]];then
        device_type="MILAN"
    else
        device_type="SPR"
    fi
    # cpu array
    numa_nodes_use_=$(( $numa_nodes_use + 1 ))

    cpu_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node//;s/cpus://" |sed -n "${numa_nodes_use_}p" |\
    awk -v cpn=${cores_per_node} '{for(i=1;i<=cpn+1;i++) {printf(" %s ",$i)} printf("\n");}' |grep '[0-9]' |\
    awk -v cpi=${cores_per_instance} -v cps=${cores_per_node} -v cores=${OOB_TOTAL_CORES_USE} '{
        if(cores == "") { cores = NF; }
        for( i=2; i<=cores; i++ ) {
            if((i-1) % cpi == 0 || (i-1) % cps == 0) {
                print $i";"$1
            }else {
                printf $i","
            }
        }
    }' |sed "s/,$//"))
    instance=${#cpu_array[@]}
    export OMP_NUM_THREADS=$(echo ${cpu_array[0]} |awk -F, '{printf("%d", NF)}')

    # environment
    gcc -v
    python -V
    pip list
    # git remote -v
    # git branch
    # git show -s
    fremework_version="$(pip list |& grep -E "^torch[[:space:]]|^tensorflow[[:space:]]" |awk '{printf("%s",$2)}')"
}

# environment
function set_environment {
    #
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
    export TF_ENABLE_ONEDNN_OPTS=1

    # AMX
    # if [ "${device_type}" == "SPR" ];then
    #     export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    # else
    #     unset DNNL_MAX_CPU_ISA
    # fi

    # NativeFormat
    if [ "${USE_TF_NATIVEFORMAT}" != "" ];then
        export TF_ENABLE_MKL_NATIVE_FORMAT=${USE_TF_NATIVEFORMAT}
    else
        unset TF_ENABLE_MKL_NATIVE_FORMAT
    fi

    # DNN Verbose
    if [ "${dnnl_verbose}" != "" ];then
        export DNNL_VERBOSE=${dnnl_verbose}
        export MKLDNN_VERBOSE=${dnnl_verbose}
    else
        unset DNNL_VERBOSE MKLDNN_VERBOSE
    fi
    # Profile
    if [ "${profile}" == "1" ];then
        addtion_options=" --profile "
    else
        addtion_options=""
    fi
}

function logs_path_clean {
    # logs saved
    log_dir="${framework}-${model_name}-${mode_name}-${precision}-bs${batch_size}-"
    log_dir+="cpi${cores_per_instance}-ins${instance}-nnu${numa_nodes_use}-$(date +'%s')"
    log_dir="${WORKSPACE}/$(echo ${log_dir} |sed 's+[^a-zA-Z0-9./-]+-+g')"
    mkdir -p ${log_dir}
    if [ ! -e ${WORKSPACE}/summary.log ];then
        printf "framework,model_name,mode_name,precision,batch_size," | tee ${WORKSPACE}/summary.log
        printf "cores_per_instance,instance,throughput,comp_time,mem_time,link,reference,benchmark_time,correctness_check\n" | tee -a ${WORKSPACE}/summary.log
    fi
    # exec cmd
    excute_cmd_file="${log_dir}/${framework}-run-$(date +'%s').sh"
    rm -f ${excute_cmd_file}
    rm -rf ./timeline
}

function collect_perf_logs {
    set +x
    comp_time=0
    mem_time=0
    # performance
    if [ $[ ${dnnl_verbose} + ${profile} ] -eq 0 ];then
        # mlpc dashboard json
        generate_json_details no_input ${log_dir}/mlpc_perf.json benchmark
    fi
    # dnnl verbose
    if [ "${dnnl_verbose}" == "1" ];then
        for i_file in $(find ${log_dir}/ -type f -name "rcpi*.log" |sort)
        do
            bash ${workload_dir}/../../one_iter_verbose.sh ${i_file} > /dev/null 2>&1 || true
            python ${workload_dir}/../../dnnl_parser.py -f ${i_file}-1iter.log |column -t >> ${log_dir}/primitive.log 2>&1 || true
            python ${workload_dir}/../../dnnl_parser.py -f ${i_file}-1iter.log --config |column -t >> ${log_dir}/shape.log 2>&1 || true
            end2end_time=$(grep 'Iteration:.*, inference time:' ${i_file}-1iter.log |tail -1 |awk '{printf("%.6f", $6 * 1000)}')
            model_name=${model_name} end2end_time=${end2end_time} log_dir=${log_dir} \
                        bash ${workload_dir}/../../fwk-flops.sh ${log_dir}/shape.log
            break
        done
        # mlpc dashboard json
        generate_json_details ${log_dir}/primitive.log ${log_dir}/mlpc_dnnl.json
    fi
    # operator
    if [ "${profile}" == "1" ];then
        mv ./timeline ${log_dir}
        for i_file in $(find ${log_dir}/timeline -type f -name "timeline*.json" |sort)
        do
            python ${workload_dir}/../../profile_parser.py -f ${i_file} |column -t >> ${log_dir}/operator.log 2>&1 || true
            break
        done
        # mlpc dashboard json
        generate_json_details ${log_dir}/operator.log ${log_dir}/mlpc_prof.json
    fi
    # get reference value
    reference_value=0
    if [ "${OOB_REFERENCE_LOG}" != "" ];then
        rm -f reference.log
        wget -q --no-proxy --no-check-certificate -O reference.log ${OOB_REFERENCE_LOG}
        reference_value=$(awk -F, -v f=${framework} -v m=${model_name} '{
            gsub (" ", "", $1);
            gsub (" ", "", $2);
            if($1 == f && $2 == m){ printf("%s", $8) }
        }' reference.log)
        rm -f reference.log
    fi
    # summary
    if [ "$BUILD_URL" != "" ];then
        artifact_url="${BUILD_URL}artifact/$(basename ${log_dir})"
    else
        artifact_url="$(basename ${log_dir})"
    fi
    # printf "${framework},${model_name},${mode_name},${precision}," |tee ${log_dir}/result.txt |tee -a ${WORKSPACE}/summary.log
    # printf "${batch_size},${cores_per_instance},${instance},${throughput}," |tee -a ${log_dir}/result.txt |tee -a ${WORKSPACE}/summary.log
    # printf "${comp_time},${mem_time},${artifact_url},${reference_value},${benchmark_time}\n" |tee -a ${log_dir}/result.txt |tee -a ${WORKSPACE}/summary.log

    printf "${framework},${model_name},${mode_name},${precision},${batch_size},${cores_per_instance},${instance},${throughput},${comp_time},${mem_time},${artifact_url},${reference_value},${benchmark_time},${correctness_check}\n" | tee -a ${log_dir}/result.txt |tee -a ${WORKSPACE}/summary.log
 
    set +x
    echo -e "\n\n-------- Summary --------"
    sed -n '1p;$p' ${WORKSPACE}/summary.log |column -t -s ','
}

function generate_json_details {
    input_log="$1"
    json_file="$2"
    if [ "$3" == "benchmark" ];then
        echo -e """
            {
                'framework' : 'TensorFlow',
                'category' : 'OOB_performance_v2',
                'device' : '${device_type}',
                'period' : '$(date +%Y)WW$(date -d 'next week'  +%U)',
                'quarter': 'Q$(date +%q),,,$(date +%y)',
                'datatype' : '${precision}',
                'results' : [
                    {
                        'bs' : '${batch_size}',
                        'framework_version' : '${fremework_version}',
                        'core/instance' : '${cores_per_instance}',
                        'comp_op' : '',
                        'cast_op' : '',
                        'model_name' : '${model_name}',
                        'perf' : '${throughput}',
                        'instance' : '${instance}'
                    }
                ]
            }
        """ |sed "s/'/\"/g;s/,,,/'/" > ${json_file}
    else
        # comp/mem time
        op_time=($(
            grep "[0-9]$" ${input_log} |awk 'BEGIN {
                comp_time = 0;
                mem_time = 0;
            } {
                if(tolower($1) ~/matmul|conv|inner_product/) {
                    comp_time += $2;
                }else {
                    mem_time += $2;
                }
            }END {
                printf("%.3f  %.3f", comp_time, mem_time);
            }'
        ))
        comp_time=${op_time[0]}
        mem_time=${op_time[1]}
        # json
        echo -e """
            {
                'framework' : 'TensorFlow',
                'category' : 'OOB_timeline',
                'device' : '${device_type}',
                'period' : '$(date +%Y)WW$(date -d 'next week'  +%U)',
                'quarter': 'Q$(date +%q),,,$(date +%y)',
                'datatype' : '${precision}',
                'results' : [
                    {
                        'model_name' : '${model_name}',
                        'cast_op': '',
                        'comp_op':'',
                        'bs' : '${batch_size}',
                        'framework_version' : '${fremework_version}',
                        'perf' : '${throughput}',
                        'instance' : '${instance}',
                        'core/instance' : '${cores_per_instance}',
                        'details': [
            $(
                grep "[0-9]$" ${input_log} |sort |awk 'BEGIN {
                    op_name = "tmp_name";
                    op_time = 0;
                    op_calls = 0;
                }{
                    if(tolower(op_name) ~/matmul|conv|inner_product/) {
                        type = "cpu";
                    }else {
                        type = "mem";
                    }
                    if($1 != op_name) {
                        if(op_name != "tmp_name") {
                            printf("{\"calls\":\"%d\", \"type\":\"%s\", \"primitive\":\"%s\", \"time_ms\":\"%.3f\"},\n", op_calls, type, op_name, op_time);
                        }
                        op_name = $1;
                        op_time = $2;
                        op_calls = $3;
                    }else {
                        op_time += $2;
                        op_calls += $3;
                    }
                }END {
                    printf("{\"calls\":\"%d\", \"type\":\"%s\", \"primitive\":\"%s\", \"time_ms\":\"%.3f\"}\n", op_calls, type, op_name, op_time);
                }'
            )
                        ]
                    }
                ]
            }
        """ |sed "s/'/\"/g;s/,,,/'/" > ${json_file}
    fi
    # post
    if [ "${OOB_MLPC_DASHBOARD}" == "1" ];then
        post_mlpc
    fi
}

function post_mlpc {
    # post data to mlpc dashboard
    mlpc_api="http://mlpc.intel.com/api/store_oob"
    curl -X POST -H "Content-Type: application/json" -d @${json_file} ${mlpc_api}
}
