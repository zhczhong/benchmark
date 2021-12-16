set -x
cd gen-efficientnet-pytorch
rm -rf logs
mkdir logs

function main {
    set -x
    model_name="alexnet,densenet121,densenet161,densenet169,densenet201,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7,efficientnet_b8,fbnetc_100,googlenet,inception_v3,mnasnet0_5,mnasnet1_0,resnet101,resnet152,resnet18,resnet34,resnet50,resnext101_32x8d,resnext50_32x4d,shufflenet_v2_x0_5,shufflenet_v2_x1_0,spnasnet_100,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,wide_resnet101_2,wide_resnet50_2"
    model_name="alexnet,densenet161,efficientnet_b2,fbnetc_100,googlenet,inception_v3,mnasnet1_0,resnet152,resnet34,resnext101_32x8d,shufflenet_v2_x0_5,spnasnet_100,squeezenet1_0,vgg16,wide_resnet50_2"
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))

    for model_name in ${model_name_list[@]}
    do
        # generate benchmark
        launch_migs ${model_name} > ./logs/temp.log
        wait
        collect_perf
    done
}

function launch_migs {
    # MIG_list=$(nvidia-smi -L | grep 'UUID' | rev | cut -d' ' -f1 | rev | cut -d')' -f1 | tail -${INSTANCES})
    set -x
    # CUDA_VISIBLE_DEVICES=MIG-fb4db3f1-449b-5f62-b2b6-885ccf1f0f6e python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-282b0b25-f327-5934-afd9-1a467f91a307 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-38f9149c-9cce-53c5-bd3a-13401f15336b python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-7839f662-c5dc-5204-a669-51c155c3de94 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-dcc281e6-86b8-57f6-880e-feb15d21e2cd python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-f5a339d5-585d-59fd-898d-0dd0ab11d5c2 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-0e98ffb7-9d48-5bc7-9cb5-873e4c20aa22 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --gpu 0
    # CUDA_VISIBLE_DEVICES=MIG-fb4db3f1-449b-5f62-b2b6-885ccf1f0f6e python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --jit_optimize --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-282b0b25-f327-5934-afd9-1a467f91a307 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --jit_optimize --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-38f9149c-9cce-53c5-bd3a-13401f15336b python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --jit_optimize --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-7839f662-c5dc-5204-a669-51c155c3de94 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --jit_optimize --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-dcc281e6-86b8-57f6-880e-feb15d21e2cd python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --jit_optimize --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-f5a339d5-585d-59fd-898d-0dd0ab11d5c2 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --jit_optimize --gpu 0 & \
    # CUDA_VISIBLE_DEVICES=MIG-0e98ffb7-9d48-5bc7-9cb5-873e4c20aa22 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --jit --jit_optimize --gpu 0
    CUDA_VISIBLE_DEVICES=MIG-fb4db3f1-449b-5f62-b2b6-885ccf1f0f6e python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --gpu 0 & \
    CUDA_VISIBLE_DEVICES=MIG-282b0b25-f327-5934-afd9-1a467f91a307 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --gpu 0 & \
    CUDA_VISIBLE_DEVICES=MIG-38f9149c-9cce-53c5-bd3a-13401f15336b python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --gpu 0 & \
    CUDA_VISIBLE_DEVICES=MIG-7839f662-c5dc-5204-a669-51c155c3de94 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --gpu 0 & \
    CUDA_VISIBLE_DEVICES=MIG-dcc281e6-86b8-57f6-880e-feb15d21e2cd python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --gpu 0 & \
    CUDA_VISIBLE_DEVICES=MIG-f5a339d5-585d-59fd-898d-0dd0ab11d5c2 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --gpu 0 & \
    CUDA_VISIBLE_DEVICES=MIG-0e98ffb7-9d48-5bc7-9cb5-873e4c20aa22 python main.py -e --performance --pretrained --dummy -w 10 -i 50 -a ${model_name} -b 1 --precision "float32" --gpu 0 	
}

function collect_perf {
    set -x
    throughput=$(grep 'inference Throughput:' ./logs/temp.log |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk '
        BEGIN {
            sum = 0;
        }
        {
            sum = sum + $1;
        }
        END {
            printf("%.3f", sum);
        }
    ')
    echo ${model_name} $throughput >> ./logs/summary_MIG.log
}

Start
main "$@"
