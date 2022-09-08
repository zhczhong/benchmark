set -x

mode_all=$1
if [ ${mode_all} == "all" ]; then
    mode_all="throughput,latency,multi_instance"
fi
mode_list=($(echo "${mode_all}" |sed 's/,/ /g'))

for mode in ${mode_list[@]}
do
    # bash run_auto_cpu.sh all float32 ${mode} ipex
    # bash run_auto_cpu.sh all float32 ${mode} torchdynamo_ipex
    #bash run_auto_cpu.sh all float32 ${mode} pt
    bash run_auto_cpu.sh all float32 ${mode} jit
    #bash run_auto_cpu.sh all float32 ${mode} jit_ofi
    # bash run_auto_cpu.sh all bfloat16 ${mode} ipex
    # bash run_auto_cpu.sh all bfloat16 ${mode} torchdynamo_ipex
    #bash run_auto_cpu.sh all bfloat16 ${mode} pt
    bash run_auto_cpu.sh all bfloat16 ${mode} jit
    #bash run_auto_cpu.sh all int8_ipex ${mode} int8_ipex

    # OpenVino
    #bash run_auto_cpu.sh all FP32 ${mode} openvino
done
