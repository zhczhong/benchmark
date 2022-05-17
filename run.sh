set -x

bash run_auto_pt_channellast.sh all float32 56
cat ./logs/summary.log
bash run_auto_pt_channellast.sh all bfloat16_brutal 56
cat ./logs/summary.log
bash run_auto_pt_channellast.sh all float32 " --fx " 56
cat ./logs/summary.log
bash run_auto_pt_channellast.sh all bfloat16_brutal " --fx " 56
cat ./logs/summary.log
bash run_auto_pt_channellast.sh all bfloat16 56
cat ./logs/summary.log
