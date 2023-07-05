# install dependency package

1. pip install package

```
pip install torchaudio --no-deps
pip install python-Levenshtein
```

2. warp-ctc
   install:

```
git clone https://github.com/SeanNaren/warp-ctc.git && \
    cd warp-ctc && \
    mkdir -p build && cd build && cmake .. && make && \
    cd ../pytorch_binding && python setup.py install
cd ../../
```

3. apply patch

```
cd PT/mlperf_training
git apply PT/workload/deepspeech/deepspeech.patch

```

# dataset

```
cd PT/mlperf_training/speech_recognition
bash ./download_dataset.sh
```

# pretrained model

put weight file at `speech_recognition/models/deepspeech_10.pth`

# run real time inference

```bash
python -u python train.py \
    --model_path models/deepspeech_10.pth \
    --seed 1 \
    --batch_size ${bs} \
    --evaluate --ipex \
    --precision float32


usage: train.py [-h] [--checkpoint] [--save_folder SAVE_FOLDER] [--model_path MODEL_PATH] [--continue_from CONTINUE_FROM] [--seed SEED] [--acc ACC]
                [--start_epoch START_EPOCH]

DeepSpeech training

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint          Enables checkpoint saving of model
  --save_folder SAVE_FOLDER
                        Location to save epoch models
  --model_path MODEL_PATH
                        Location to save best validation model
  --continue_from CONTINUE_FROM
                        Continue from checkpoint model
  --seed SEED           Random Seed
  --acc ACC             Target WER
  --start_epoch START_EPOCH
                        Number of epochs at which to start from

```
