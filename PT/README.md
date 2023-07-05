# PT models benchmarking

Inference (real time) scripts with Pytorch & intel-extension-for-pytorch.

## Prerequisites

```
torch            1.5.0a0+b58f89b
torch-ipex       1.0.0
torchvision      0.6.0
```

For torch/torch-ipex install, please refer to: [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch).

## Benchmarking

For "efficientnet-b0" ~ "efficientnet-b8", "efficientnet-l2", and all torchvision models benchmark, please refer to [gen-efficientnet-pytorch
](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/PT%2Fworkload%2Fgen-efficientnet-pytorch), other info please see below table.

```shell
    pip install -r requirements.txt
    cd gen-efficientnet-pytorch
    git apply ../workload/gen-efficientnet-pytorch/gen-efficientnet-pytorch.diff
    python setup.py install

    python -u ./main.py ${DATASET_PATH} \
                        -e --performance \
                        --pretrained \
                        --no-cuda \
                        --mkldnn \
                        -j 1  \
                        -w 10 \
                        -b 1 \
                        -i 100 \
                        -a ${MODEL_NAME} \
                        --dummy --jit

```


|   | Model | Repo | Base Commit | model_path | comments |
| - | - | - | - | - | - |
| OOB-20-10 | alexnet | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | resnet18 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | resnet34 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | resnet50 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | resnet101 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | resnet152 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | squeezenet1_0 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | squeezenet1_1 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg11 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg13 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg16 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg19 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg11_bn | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg13_bn | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg16_bn | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | vgg19_bn | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | shufflenet_v2_x0_5 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | shufflenet_v2_x1_0 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | googlenet | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | resnext50_32x4d | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | resnext101_32x8d | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | wide_resnet50_2 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | wide_resnet101_2 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | inception_v3 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | efficientnet_b0 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | efficientnet_b5 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | efficientnet_b7 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | mnasnet1_0 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | mnasnet0_5 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | densenet121 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | densenet161 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | densenet169 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |
| OOB-20-10 | densenet201 | https://github.com/rwightman/gen-efficientnet-pytorch | 8795d329 |   |   |

For inceptionresnetv2, se_resnext50_32x4d, vggm, please refer to: [pretrained-models](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/PT%2Fworkload%2Fpretrained-models), other info please see below table.

```shell
    cd pretrained-models
    git apply ../workload/pretrained-models/vggm.diff
    python setup.py install
  
    python -u examples/imagenet_eval.py --data ${DATASET_DIR} \
                                        -e --performance \
                                        --pretrained None \
                                        --mkldnn \
                                        -j 1 \
                                        -b 1 \
                                        -i 100 \
                                        -a ${MODEL_PATH} \
                                        --dummy \

```


|   | Model | Repo | Base Commit | model_path | comments |
| - | - | - | - | - | - |
| OOB-20-10 | inceptionresnetv2 | https://github.com/Cadene/pretrained-models.pytorch | 8aae3d8 |   |   |
| OOB-20-10 | se_resnext50_32x4d | https://github.com/Cadene/pretrained-models.pytorch | 8aae3d8 |   |   |
| OOB-20-10 | vggm | https://github.com/Cadene/pretrained-models.pytorch | 8aae3d8 |   |   |

For other OOB Q1/Q2 models, please refer to related README (WIP), other info please see below table.


|   | Model | Repo | Base Commit | model_path | comments |
| - | - | - | - | - | - |
| OOB-20-10 | 3D-UNet | https://github.com/wolny/pytorch-3dunet | 88bf7ecd |   |   |
| OOB-20-10 | CRNN | https://github.com/meijieru/crnn.pytorch | d3a47f91 |   |   |
| OOB-20-10 | SSD300 | https://github.com/PenghuiCheng/training/tree/master/single_stage_detector/ssd | TBD |   |   |
| OOB-20-10 | MaskRCNN | https://github.com/PenghuiCheng/training/tree/master/object_detection | TBD |   |   |
| OOB-20-10 | RetinaNet | https://github.com/facebookresearch/detectron2 | TBD |   |   |
| OOB-20-10 | BERT-LARGE | https://github.com/huggingface/transformers | 7972a401 |   |   |
| OOB-20-10 | Transformer-LT | https://github.com/pytorch/fairseq/tree/master/examples/translation | TBD |   |   |
| OOB-20-10 | Convolution Seq2Seq | https://github.com/pytorch/fairseq/tree/master/examples/conv_seq2seq | TBD |   |   |
| OOB-20-10 | GPT-2 | https://github.com/huggingface/transformers/blob/master/examples/run_generation.py | 7972a401 |   |   |
| OOB-20-10 | RNN-T | https://github.com/mlperf/inference.git | db0b7eb3 |   |   |
| OOB-20-10 | DCGAN | https://github.com/pytorch/examples/tree/master/dcgan | 391be73b |   |   |
| OOB-20-10 | W&D | https://github.com/jrzaurin/pytorch-widedeep | 7f4582e8 |   |   |
| OOB-20-10 | ResNext3D | https://github.com/facebookresearch/ClassyVision | TBD |   |   |
| OOB-20-10 | Facenet | https://github.com/timesler/facenet-pytorch | 5be876e3 |   |   |
| OOB-20-10 | TTS | https://github.com/mozilla/TTS | fab74dd5 |   |   |
| OOB-20-10 | XLNet-Classification | https://github.com/huggingface/transformers/blob/master/examples/run_glue.py | 7972a401 |   |   |
