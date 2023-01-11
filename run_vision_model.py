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

vision_model_list = [
    'alexnet', 'dcgan', 'densenet121', 'functorch_dp_cifar10',
    'functorch_maml_omniglot', 'maml_omniglot', 'mnasnet1_0', 'mobilenet_v2',
    'mobilenet_v3_large', 'phlippe_densenet', 'phlippe_resnet',
    'pytorch_CycleGAN_and_pix2pix', 'pytorch_stargan', 'resnet18', 'resnet50',
    'resnet152', 'resnext50_32x4d', 'shufflenet_v2_x1_0', 'squeezenet1_1',
    'timm_efficientnet', 'timm_regnet', 'timm_vision_transformer',
    'timm_vision_transformer_large', "vgg16"
]
vision_model_list = ["alexnet"]


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
    name = model_name.replace(" ", "_") + "_" + family + "_" + stepping
    return name


def run(args):
    dataframe = pd.DataFrame(
        columns=["model", "backend", "batch_size", "time/ms", "throughput"])
    bench_cmd = [
        "python", "-m", "intel_extension_for_pytorch.cpu.launch",
        "--use_default_allocator", "--throughput_mode", "--benchmark",
        "--ninstances", "1", "run.py", "-t", "eval", "-m", "jit", "-d", "cpu",
        "--datatypes={}".format(args.datatypes)
    ]

    model_list = vision_model_list if args.model == "all" else [args.model]

    for model_name in vision_model_list:
        cmd = bench_cmd + [model_name]
        new_row = {}
        with subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              bufsize=1,
                              universal_newlines=True) as p:
            print(" ".join(cmd))
            time = "failed"
            throughput = "failed"
            correctness = "false"
            batch_size = "unknown"
            for out_line in p.stdout:
                print(out_line)
                if "CPU Total Wall Time" in out_line:
                    time = re.findall("\d+.\d+", out_line)[0].strip(' ')
                if "Throughtput" in out_line:
                    throughput = re.findall("\d+.\d+", out_line)[0].strip(' ')
                if "batch size" in out_line:
                    batch_size = re.findall("\d+", out_line)[0].strip(' ')
            new_row["model"] = model_name + "_bs" + batch_size
            new_row["backend"] = args.backend
            new_row["batch_size"] = batch_size
            new_row["time/ms"] = time
            new_row["throughput"] = throughput
            print(new_row.values())
            dataframe.loc[len(dataframe.index)] = new_row.values()
    print(dataframe)
    dataframe.to_csv(args.output_path)


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description="Vision Model Performance Evaluation")
    parser.add_argument('--backend',
                        type=str,
                        default="gc+dnnl",
                        choices=['dnnl', 'gc', 'ipex', 'pytorch', 'gc+dnnl'])
    parser.add_argument("--model",
                        type=str,
                        default="all",
                        choices=vision_model_list)
    parser.add_argument("--output_path", type=str, default="default")
    parser.add_argument("--datatypes",
                        type=str,
                        default="f32",
                        choices=["f32", "bf16"])
    args = parser.parse_args()

    if args.output_path == "default":
        args.output_path = "./" + get_code_name() + "_" + args.backend + \
            "_" + args.datatypes + "_" + "report.csv"

    cpu_cores = get_cpu_cores()
    is_amx = is_running_on_amx()
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    os.environ[
        "LD_PRELOAD"] = "$HOME/ipex_env/miniconda/envs/ipex_env/lib/libiomp5.so:$HOME/ipex_env/miniconda/envs/ipex_env/lib/libjemalloc.so"
    os.environ[
        "DNNL_MAX_CPU_ISA"] = "AVX512_CORE_AMX" if is_amx else "AVX512_CORE_VNNI"
    os.environ[
        "MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    os.environ["_DNNL_GRAPH_FORCE_MAX_PARTITION_POLICY"] = "1"
    os.environ[
        "_DNNL_GRAPH_DISABLE_DNNL_BACKEND"] = "0" if "dnnl" in args.backend.split(
            "+") else "1"
    os.environ[
        "_DNNL_GRAPH_DISABLE_COMPILER_BACKEND"] = "0" if "gc" in args.backend.split(
            "+") else "1"
    run(args)


if __name__ == "__main__":
    main()
