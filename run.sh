set -x

bash run_auto_cpu.sh all float32 throughput ipex
# bash run_auto_cpu.sh all float32 throughput torchdynamo_ipex
bash run_auto_cpu.sh all float32 throughput pt
bash run_auto_cpu.sh all float32 throughput ofi
bash run_auto_cpu.sh all bfloat16 throughput ipex
# bash run_auto_cpu.sh all bfloat16 throughput torchdynamo_ipex
bash run_auto_cpu.sh all bfloat16 throughput pt
bash run_auto_cpu.sh all bfloat16 throughput ofi
bash run_auto_cpu.sh all int8_ipex throughput int8_ipex
