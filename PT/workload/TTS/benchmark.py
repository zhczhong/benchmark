# Based on Jupyter notebook from:
# https://github.com/mozilla/TTS/blob/20a6ab3d612eea849a49801d365fcd071839b7a1/notebooks/Benchmark.ipynb

import os
import sys
import io
import torch
import time
import json
import numpy as np
from collections import OrderedDict
from matplotlib import pylab as plt

import librosa
import librosa.display

from TTS.models.tacotron import Tacotron
from TTS.layers import *
from TTS.utils.data import *
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text import text_to_sequence
from TTS.utils.synthesis import synthesis, text_to_seqvec, id_to_torch
from TTS.utils.visual import visualize
from TTS.utils.text.symbols import symbols, phonemes

import argparse
import csv

def main():
    args = parse_args()
    if args.ipex:
        import intel_pytorch_extension as ipex
        ipex.enable_auto_optimization()
    # Set constants
    model_path = args.model_path
    config_path = args.config_path
    config = load_config(config_path)
    dataset_name = args.dataset_name
    metadata_path = args.metadata_path
    test_sentences = load_test_sentences(dataset_name, metadata_path)
    use_cuda = False
    # Set some config fields manually for testing
    config.use_forward_attn = True
    # Use only one speaker
    speaker_id = None
    speakers = []
    # load the audio processor
    ap = AudioProcessor(**config.audio)
    # Load TTS model
    model = load_model(config, speakers, model_path)
    # if args.jit:
    #     for i, sentence in enumerate(test_sentences):
    #         inputs = text_to_seqvec(sentence, config, use_cuda)
    #         speaker_id_ = id_to_torch(speaker_id)
    #         model = torch.jit.trace(model.inference, (inputs, speaker_id_))
    #         break
    if args.ipex:
        model.to(ipex.DEVICE)
    # Run inference
    avg_throughput, avg_latency = run_inference(test_sentences, model, config, use_cuda, ap, speaker_id, device="dpcpp" if args.ipex else "cpu")
    print(f"Total Average Throughput: {avg_throughput} [element/s]")
    print(f"Total Average Latency: {avg_latency} [ms/element]")

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model-path", help="Tacotron model filename", action="store", required=True)
    parser.add_argument("--config-path", help="Tacotron config filename", action="store", required=True)
    parser.add_argument("--dataset-name", help="Dataset name", action="store", default="dummy", required=False)
    parser.add_argument("--metadata-path", help="Dataset metadata filename (file containing sentences)",
                        default="./dummy_data.csv", action="store", required=False)
    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    args = parser.parse_args()
    return args

def load_test_sentences(dataset_name, metadata_path):
    dataset_load_function = {
        "dummy": get_dummy_data,
        "ljspeech": get_ljspeech_data
    }
    with open(metadata_path) as filestream:
        raw_data = filestream.readlines()
    return dataset_load_function.get(dataset_name, lambda x: [])(raw_data)

def get_dummy_data(raw_data):
    reader = csv.reader(raw_data)
    data = [element for row in reader for element in row if element]
    return data

def get_ljspeech_data(raw_data):
    reader = csv.reader(raw_data, delimiter="|", quoting=csv.QUOTE_NONE)
    data = [row[2] for row in reader]
    return data

def load_model(config, speakers, model_path):
    # Only one speaker
    speakers = []
    # load the model
    num_chars = len(phonemes) if config.use_phonemes else len(symbols)
    model = setup_model(num_chars, len(speakers), config)
    # load model state
    cp = torch.load(model_path, map_location=lambda storage, loc: storage)
    # load the model
    model.load_state_dict(cp['model'])
    model.eval()
    # set model stepsize
    if 'r' in cp:
        model.decoder.set_r(cp['r'])
    model.decoder.max_decoder_steps = 2000
    return model

def run_inference(test_sentences, model, config, use_cuda, ap, speaker_id, device="cpu"):
    num_sentences = len(test_sentences)
    throughputs = []
    latencies = []
    avg_throughput = 0
    avg_latency = 0
    for i, sentence in enumerate(test_sentences):
        write_progress_bar(i, num_sentences, avg_throughput, avg_latency)
        throughput_s, latency_ms = tts(model, sentence, config, use_cuda, ap, speaker_id, figures=True, device=device)
        throughputs.append(throughput_s)
        latencies.append(latency_ms)
        avg_throughput = sum(throughputs)/len(throughputs)
        avg_latency = sum(latencies)/len(latencies)
    close_progress_bar()
    return avg_throughput, avg_latency

def write_progress_bar(i, iterations, avg_throughput, avg_latency, toolbar_width=50):
    num_bars = round(toolbar_width*i/iterations)
    line = ["[", u"\u2588"*num_bars, " "*(toolbar_width-num_bars), "]",
            " ", str(i), "/", str(iterations),
            " ", "|", " ", "avg thrpt: ", f"{avg_throughput:.2f}", " [elem/s]",
            " ", "|", " ", "avg lat: ", f"{avg_latency:.2f}", " [ms/elem]"]
    line_str = "".join(line)
    line_width = len(line_str)
    sys.stdout.write(line_str)
    sys.stdout.flush()
    sys.stdout.write("\b" * line_width)
    sys.stdout.flush()

def close_progress_bar():
    sys.stdout.write("\n")
    sys.stdout.flush()

def tts(model, text, CONFIG, use_cuda, ap, speaker_id, figures=True, device="cpu"):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs_size = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, False, CONFIG.enable_eos_bos_chars, device=device)
    mel_postnet_spec = ap._denormalize(mel_postnet_spec)
    mel_postnet_spec = ap._normalize(mel_postnet_spec)
    run_time = time.time() - t_1
    latency_s = run_time / inputs_size[1]
    latency_ms = latency_s * 1000
    throughput_s = 1 / latency_s
    return throughput_s, latency_ms

if __name__ == "__main__":
    sys.exit(main())
