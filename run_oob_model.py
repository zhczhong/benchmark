################################################################################
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import argparse
import csv
import os
import subprocess
import re
import random
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torchvision as vision
import timm

torch_bench_vision_model_list = ['dcgan', 'phlippe_densenet', 'phlippe_resnet']

torch_vision_classification_model_list = [
    "alexnet", "convnext_tiny", "convnext_small", "convnext_base",
    "convnext_large", "densenet121", "densenet161", "densenet169",
    "densenet201", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6",
    "efficientnet_b7", "googlenet", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0",
    "mnasnet1_3", "inception_v3", "mobilenet_v2", "mobilenet_v3_large",
    "mobilenet_v3_small", "resnet18", "resnet34", "resnet50", "resnet101",
    "resnet152", 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf',
    'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf',
    'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf',
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnext101_32x8d',
    'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
    'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2',
    'wide_resnet50_2'
]  # image_size 224, skip: "efficientnet_v2_s", "efficientnet_v2_m","efficientnet_v2_l", "'regnet_y_128gf'", 'vit_b_16','vit_b_32', 'vit_l_16', 'vit_l_32', 'swin_b', 'swin_s','swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t'

torch_vision_segmentation_model_list = [
    'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101',
    'deeplabv3_resnet50', 'fcn_resnet101', 'fcn_resnet50',
    "lraspp_mobilenet_v3_large"
]  # image_size 520

VIT_model_list = ["DeepViT"]

timm_vision_model_list = [
    "fbnetc_100", "spnasnet_100", "vit_small_patch16_224",
    "vit_giant_patch14_224", "regnety_120", "resnest14d", "efficientnet_b0"
]
vision_model_list = {
    "torchvision": torch_vision_classification_model_list,
    "timm": timm_vision_model_list,
    "torchbench": torch_bench_vision_model_list
}


def get_cpu_cores():
    with subprocess.Popen(["lscpu"],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          bufsize=1,
                          universal_newlines=True) as p:
        for line in p.stdout:
            if "NUMA node0 CPU(s):" in line:
                cores = int(re.findall("(?<=-)[0-9]*", line)[0])
                return cores + 1
    return 56


def is_running_on_amx():
    with subprocess.Popen(["lscpu"],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          bufsize=1,
                          universal_newlines=True) as p:
        for line in p.stdout:
            if "amx_tile" in line:
                return True
        return False


def get_code_name():
    model_name = "unknown"
    family = "unknown"
    stepping = "unknown"

    with subprocess.Popen(["lscpu"],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          bufsize=1,
                          universal_newlines=True) as p:
        for line in p.stdout:
            if "Model name" in line:
                model_name = line.split(":")[1].lstrip().rstrip("\n")
            if "CPU family" in line:
                family = line.split(":")[1].lstrip().rstrip("\n")
            if "Stepping" in line:
                stepping = line.split(":")[1].lstrip().rstrip("\n")
    name = model_name.replace(" ", "_").replace(
        "(R)", "") + "_" + family + "_" + stepping
    return name


def run(args):
    dataframe = pd.DataFrame(columns=[
        "model", "model_source", "batch_size", "core_per_instance",
        "datatypes", "throughput_per_instance(GC)", "Correctness(GC)",
        "throughput_per_instance(DNNL)", "Correctness(DNNL)"
    ])

    cpu_cores = get_cpu_cores()
    datatypes = ["f32", "bfloat16", "int8"]
    for core_per_instance_bs_pair in [(4, 1), (cpu_cores, 112)]:
        core_per_instance = core_per_instance_bs_pair[0]
        bs = core_per_instance_bs_pair[1]
        num_instance = cpu_cores // core_per_instance
        os.environ["OMP_NUM_THREADS"] = str(core_per_instance)
        for dtype in datatypes:
            for model_source in ["torchvision", "timm", "torchbench"]:
                for model_name in vision_model_list[model_source]:
                    bench_cmd = "python -m intel_extension_for_pytorch.cpu.launch --use_default_allocator --ninstance=2 --benchmark main.py -e --performance --pretrained -j 1 -w 20 -b {batch_size} -i 200 -a {model_name} --dummy --precision={data_type} --llga --model-source={model_source} --weight-sharing --number-instance={number_instance} --check_correctness".format(
                        batch_size=bs,
                        model_name=model_name,
                        data_type=dtype,
                        model_source=model_source,
                        number_instance=num_instance)
                    cmd = bench_cmd.split(" ")
                    new_row = {}
                    new_row["model"] = model_name
                    new_row["model_source"] = model_source
                    new_row["batch_size"] = bs
                    new_row["core_per_instance"] = core_per_instance
                    new_row["datatypes"] = dtype
                    os.environ["_DNNL_GRAPH_DISABLE_COMPILER_BACKEND"] = "0"
                    if len(args.dump_graph) > 0:
                        path_name = args.dump_graph + \
                            "/{dtype}/{model_source}/{model_name}/bs{bs}/".format(dtype=dtype, model_source=model_source,
                                model_name=model_name, bs=bs)
                        if not os.path.exists(path_name):
                            os.makedirs(path_name)
                        os.environ[
                            "ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GRAPH_JSON"] = path_name
                    with subprocess.Popen(cmd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          bufsize=1,
                                          universal_newlines=True) as p:
                        print(" ".join(cmd))
                        throughput = "failed"
                        correctness = "fail"
                        for out_line in p.stdout:
                            print(out_line)
                            if "inference throughput on master instance" in out_line:
                                throughput = re.findall("\d+.\d+",
                                                        out_line)[0].strip(' ')
                            if "Correctness result" in out_line:
                                correctness = out_line.split(":")[-1].strip(
                                    "\n")
                        new_row["throughput_per_instance(GC)"] = throughput
                        new_row["Correctness(GC)"] = correctness

                    os.environ["_DNNL_GRAPH_DISABLE_COMPILER_BACKEND"] = "1"
                    with subprocess.Popen(cmd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          bufsize=1,
                                          universal_newlines=True) as p:
                        print(" ".join(cmd))
                        time = "failed"
                        throughput = "failed"
                        for out_line in p.stdout:
                            print(out_line)
                            if "inference throughput on master instance" in out_line:
                                throughput = re.findall("\d+.\d+",
                                                        out_line)[0].strip(' ')
                            if "Correctness result" in out_line:
                                correctness = out_line.split(":")[-1].strip(
                                    "\n")
                        new_row["throughput_per_instance(DNNL)"] = throughput
                        new_row["Correctness(DNNL)"] = correctness
                        print(new_row.values())
                    dataframe.loc[len(dataframe.index)] = new_row.values()
                    dataframe.to_csv(args.output_path)
    print(dataframe)
    dataframe.to_csv(args.output_path)


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description="Vision Model Performance Evaluation")
    parser.add_argument("--model",
                        type=str,
                        default="all",
                        choices=vision_model_list)
    parser.add_argument("--output_path", type=str, default="default")
    parser.add_argument("--dump_graph", type=str, default="")
    args = parser.parse_args()

    if args.output_path == "default":
        args.output_path = "./" + get_code_name() + "_" + "report.csv"

    is_amx = is_running_on_amx()
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ[
        "LD_PRELOAD"] = "$HOME/ipex_env/miniconda/envs/ipex_env/lib/libiomp5.so:$HOME/ipex_env/miniconda/envs/ipex_env/lib/libjemalloc.so"
    os.environ[
        "DNNL_MAX_CPU_ISA"] = "AVX512_CORE_AMX" if is_amx else "AVX512_CORE_VNNI"
    os.environ[
        "MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    os.environ["_DNNL_GRAPH_FORCE_MAX_PARTITION_POLICY"] = "1"
    run(args)


if __name__ == "__main__":
    main()
