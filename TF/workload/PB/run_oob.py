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
from multiprocessing import Pool

supported_datatypes = ["f32", "bfloat16", "int8", "all"]
supported_scenarios = ["realtime", "throughput", "all"]

oob_model_list = [
    "aipg-vdcnn",
    "arttrack-coco-multi",
    "arttrack-mpii-single",
    "ava-face-recognition-3_0_0",
    "ava-person-vehicle-detection-stage2-2_0_0",
    "cpm-person",
    "ctpn",
    "darknet19",
    "darknet53",
    "DeepLab",
    "deeplabv3",
    "deepvariant_wgs",
    "densenet-121",
    "densenet-161",
    "densenet-169",
    "dilation",
    "DSSD_12",
    "east_resnet_v1_50",
    "efficientnet-b0",
    "efficientnet-b0_auto_aug",
    "efficientnet-b5",
    "efficientnet-b7_auto_aug",
    "faster_rcnn_inception_resnet_v2_atrous_coco",
    "faster_rcnn_inception_v2_coco",
    "faster_rcnn_nas_coco_2018_01_28",
    "faster_rcnn_nas_lowproposals_coco",
    "faster_rcnn_resnet101_ava_v2_1",
    "faster_rcnn_resnet101_coco",
    "faster_rcnn_resnet101_kitti",
    "faster_rcnn_resnet101_lowproposals_coco",
    "faster_rcnn_resnet50_coco",
    "faster_rcnn_resnet50_lowproposals_coco",
    "GAN",
    "gmcnn-places2",
    "googlenet-v1",
    "googlenet-v2",
    "googlenet-v3",
    "googlenet-v4",
    "GraphSage",
    "handwritten-score-recognition-0003",
    "i3d-flow",
    "i3d-rgb",
    "icnet-camvid-ava-0001",
    "icnet-camvid-ava-sparse-30-0001",
    "icnet-camvid-ava-sparse-60-0001",
    "icv-emotions-recognition-0002",
    "image-retrieval-0001",
    "inception-resnet-v2",
    "inceptionv2_ssd",
    "intel-labs-nonlocal-dehazing",
    "learning-to-see-in-the-dark-fuji",
    "learning-to-see-in-the-dark-sony",
    "license-plate-recognition-barrier-0007",
    "mask_rcnn_inception_resnet_v2_atrous_coco",
    "openpose-pose",
    "optical_character_recognition-text_recognition-tf",
    "person-vehicle-bike-detection-crossroad-yolov3-1020",
    "PRNet",
    "resnet-101",
    "resnet-152",
    "resnet-50",
    "resnet-v2-101",
    "resnet-v2-152",
    "resnet-v2-50",
    "retinanet",
    "R-FCN",
    "rfcn-resnet101-coco",
    "rmnet_ssd",
    "squeezenet1_1",
    "ssd_inception_v2_coco",
    "ssd_resnet34_300x300",
    "SSD_ResNet50_V1_FPN_640x640_RetinaNet50",
    "ssd_resnet50_v1_fpn_coco",
    "TCN",
    "text-recognition-0012",
    "tiny_yolo_v1",
    "tiny_yolo_v2",
    "vehicle-attributes-barrier-0103",
    "vehicle-license-plate-detection-barrier-0123",
    "vgg16",
    "vgg19",
    "vggvox",
    "Vnet",
    "WGAN",
    "yolo-v2",
    "yolo-v2-ava-sparse-35-0001",
    "yolo-v2-ava-sparse-70-0001",
    "yolo-v2-tiny-ava-0001",
    "yolo-v2-tiny-ava-sparse-30-0001",
    "yolo-v2-tiny-ava-sparse-60-0001",
    "yolo-v2-tiny-vehicle-detection-0001",
    "yolo-v3",
    "yolo-v3-tiny"
    # "3d-pose-baseline",
    # "bert-base-uncased_L-12_H-768_A-12",
    # "DRAW",
    # "Hierarchical_LSTM",
    # "HugeCTR",
    # "NCF-1B",
    # "Transformer-LT",
]
slected_model = [
    "faster_rcnn_inception_resnet_v2_atrous_coco",
    "faster_rcnn_nas_coco_2018_01_28",
    "faster_rcnn_nas_lowproposals_coco",
    "faster_rcnn_resnet101_ava_v2_1",
    "faster_rcnn_resnet101_coco",
    "faster_rcnn_resnet101_kitti",
    "faster_rcnn_resnet101_lowproposals_coco",
    "faster_rcnn_resnet50_coco",
    "faster_rcnn_resnet50_lowproposals_coco",
    "GAN",
    "icnet-camvid-ava-0001",
    "icnet-camvid-ava-sparse-30-0001",
    "icnet-camvid-ava-sparse-60-0001",
    "image-retrieval-0001",
    "intel-labs-nonlocal-dehazing",
    "mask_rcnn_inception_resnet_v2_atrous_coco",
    "openpose-pose",
    "retinanet",
    "R-FCN",
    "rfcn-resnet101-coco",
    "rmnet_ssd",
    "SSD_ResNet50_V1_FPN_640x640_RetinaNet50",
    "ssd_resnet50_v1_fpn_coco",
    "vehicle-license-plate-detection-barrier-0123",
    "WGAN",
    "yolo-v2-ava-sparse-35-0001",
    "yolo-v2-ava-sparse-70-0001",
    "yolo-v2-tiny-ava-0001",
    "yolo-v2-tiny-ava-sparse-30-0001",
    "yolo-v2-tiny-ava-sparse-60-0001",
]
# oob_model_list = slected_model


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
    cpu_cores = get_cpu_cores()
    datatypes = ["int8", "bfloat16", "f32" 
                 ] if args.data_type == "all" else [args.data_type]
    scenarios = ["realtime", "throughput"
                 ] if args.scenario == "all" else [args.scenario]
    core_bs_scenario_map = {
        "realtime": (4, 1),
        "throughput": (cpu_cores, cpu_cores * 2)
    }

    target_model = args.model
    model_list = [target_model
                  ] if target_model in oob_model_list else oob_model_list
    target_model = "all" if target_model in oob_model_list else args.model
    for dtype in datatypes:
        for scenario in scenarios:
            core_per_instance_bs_pair = core_bs_scenario_map[scenario]
            core_per_instance = core_per_instance_bs_pair[0]
            bs = core_per_instance_bs_pair[1]
            for option in [1]:
                if option == 1:
                    # GC Generic Paattern
                    os.environ["ITEX_ONEDNN_GRAPH"] = "1"
                    os.environ["ITEX_NATIVE_FORMAT"] = "1"
                    os.environ["ITEX_LAYOUT_OPT"] = "0"
                    os.environ["_DNNL_DISABLE_COMPILER_BACKEND"] = "0"
                    os.environ["_DNNL_FORCE_MAX_PARTITION_POLICY"] = "0"
                elif option == 2:
                    # GC Max partition
                    os.environ["ITEX_ONEDNN_GRAPH"] = "1"
                    os.environ["ITEX_NATIVE_FORMAT"] = "1"
                    os.environ["ITEX_LAYOUT_OPT"] = "0"
                    os.environ["_DNNL_DISABLE_COMPILER_BACKEND"] = "0"
                    os.environ["_DNNL_FORCE_MAX_PARTITION_POLICY"] = "1"
                elif option == 3:
                    # DNNL
                    os.environ["ITEX_ONEDNN_GRAPH"] = "1"
                    os.environ["ITEX_NATIVE_FORMAT"] = "1"
                    os.environ["ITEX_LAYOUT_OPT"] = "0"
                    os.environ["_DNNL_DISABLE_COMPILER_BACKEND"] = "1"
                    os.environ["_DNNL_FORCE_MAX_PARTITION_POLICY"] = "0"
                elif option == 4:
                    os.environ["ITEX_ONEDNN_GRAPH"] = "0"
                if option != 4:
                    os.environ["_ITEX_ONEDNN_GRAPH_ALL_TYPE"] = "0" if dtype == "int8" else "1"
                for model_name in model_list:
                    if target_model == "all" or target_model == model_name:
                        bench_cmd = "timeout 15m ./launch_benchmark.sh --framework=tensorflow --model_name={model_name} --mode_name=throughput --precision={data_type} --batch_size={batch_size} --numa_nodes_use=0 --cores_per_instance={core_per_instance} --num_warmup=10 --num_iter=50 --channels_last=1 --profile=0 --dnnl_verbose=0".format(
                            batch_size=bs,
                            model_name=model_name,
                            data_type=dtype,
                            core_per_instance=core_per_instance)
                        if len(args.dump_graph) > 0:
                            path_name = args.dump_graph + \
                                "/{dtype}/{model_name}/bs{bs}/".format(dtype=dtype,
                                    model_name=model_name, bs=bs)
                            if not os.path.exists(path_name):
                                os.makedirs(path_name)
                            os.environ[
                                "ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GRAPH_JSON"] = path_name
                        with subprocess.Popen(bench_cmd.split(" "),
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE,
                                              bufsize=1,
                                              universal_newlines=True) as p:
                            print(bench_cmd)
                            for out_line in p.stdout:
                                print(out_line)


# def run(args):
#     cpu_cores = get_cpu_cores()
#     datatypes = ["f32", "bfloat16", "int8"
#                  ] if args.data_type == "all" else [args.data_type]
#     scenarios = ["realtime", "throughput"
#                  ] if args.scenario == "all" else [args.scenario]
#     core_bs_scenario_map = {
#         "realtime": (4, 1),
#         "throughput": (cpu_cores, cpu_cores * 2)
#     }

#     target_model = args.model
#     model_list = [target_model
#                   ] if target_model in oob_model_list else oob_model_list
#     target_model = "all" if target_model in oob_model_list else args.model
#     f = open("run_oob_cmd.sh", "w")
#     for scenario in scenarios:
#         core_per_instance_bs_pair = core_bs_scenario_map[scenario]
#         core_per_instance = core_per_instance_bs_pair[0]
#         bs = core_per_instance_bs_pair[1]
#         for is_not_compiler_backend in ["0", "1"]:
#             for dtype in datatypes:
#                 for model_name in model_list:
#                     if target_model == "all" or target_model == model_name:
#                         bench_cmd = "./launch_benchmark.sh --framework=tensorflow --model_name={model_name} --mode_name=throughput --precision={data_type} --batch_size={batch_size} --numa_nodes_use=1 --cores_per_instance={core_per_instance} --num_warmup=20 --num_iter=200 --channels_last=1 --profile=0 --dnnl_verbose=0".format(
#                             batch_size=bs,
#                             model_name=model_name,
#                             data_type=dtype,
#                             core_per_instance=core_per_instance)
#                         path_name = "\"\""
#                         if len(args.dump_graph) > 0:
#                             path_name = args.dump_graph + \
#                                 "/{dtype}/{model_source}/{model_name}/bs{bs}/".format(dtype=dtype, model_source=model_source,
#                                     model_name=model_name, bs=bs)
#                             if not os.path.exists(path_name):
#                                 os.makedirs(path_name)

#                         env_str = "ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GRAPH_JSON={dump_path} KMP_AFFINITY=\"granularity=fine,compact,1,0\" KMP_BLOCKTIME=1 DNNL_MAX_CPU_ISA={ISA} MALLOC_CONF=\"oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000\" _DNNL_FORCE_MAX_PARTITION_POLICY=1 _DNNL_DISABLE_COMPILER_BACKEND={backend} ".format(
#                             dump_path=path_name,
#                             ISA="AVX512_CORE_AMX"
#                             if is_running_on_amx() else "AVX512_CORE_VNNI",
#                             backend=is_not_compiler_backend)
#                         f.write(env_str + bench_cmd)
#                         f.write("\n")
#     f.close()


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description="Vision Model Performance Evaluation")
    parser.add_argument("--model",
                        type=str,
                        default="all",
                        choices=oob_model_list)
    parser.add_argument("--dump_graph", type=str, default="")
    parser.add_argument("--data_type",
                        type=str,
                        default="all",
                        choices=supported_datatypes)
    parser.add_argument("--scenario",
                        type=str,
                        default="all",
                        choices=supported_scenarios)
    args = parser.parse_args()

    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "1"
    # os.environ[
    #     "LD_PRELOAD"] = "$HOME/ipex_env/miniconda/envs/ipex_env/lib/libiomp5.so:$HOME/ipex_env/miniconda/envs/ipex_env/lib/libjemalloc.so:$LD_PRELOAD"
    is_amx = is_running_on_amx()
    os.environ["_ONEDNN_CONSTANT_CACHE"] = "1"
    os.environ[
        "DNNL_MAX_CPU_ISA"] = "AVX512_CORE_AMX" if is_amx else "AVX512_CORE_VNNI"
    os.environ[
        "MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    run(args)


if __name__ == "__main__":
    main()
