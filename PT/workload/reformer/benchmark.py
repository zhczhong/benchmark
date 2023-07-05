from reformer_pytorch import Reformer

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')
    parser.add_argument('--ipex', action='store_true', default=False,
                       help='use ipex')
    # parser.add_argument('--jit', action='store_true', default=False,
    #                     help='use ipex')
    parser.add_argument('--precision', default="float32",
                            help='precision, "float32" or "bfloat16"')
    parser.add_argument('--warmup', type=int, default=2,
                        help='number of warmup')
    parser.add_argument('--max_iters', type=int, default=10,
                        help='max number of iterations to run')

    args=parser.parse_args()
    return args


# instantiate model

# model = ReformerLM(
#     dim = 512,
#     depth = 6,
#     max_seq_len = SEQ_LEN,
#     num_tokens = 256,
#     heads = 8,
#     bucket_size = 64,
#     n_hashes = 4,
#     ff_chunks = 10,
#     lsh_dropout = 0.1,
#     weight_tie = True,
#     causal = True,
#     n_local_attn_heads = 4,
#     use_full_attn = False # set this to true for comparison with full attention
# )

# model = TrainingWrapper(model)
print("create model...")
model = Reformer(
    dim = 512,
    depth = 12,
    max_seq_len = 4096,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
)
# model.cuda()

# prepare enwik8 data

def create_dataset():
    with gzip.open('./data/enwik8.gz') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = 0 if self.count == 0 else self.sum / self.count
        return avg
# setup deepspeed

cmd_args = add_argument()
input = torch.randn(cmd_args.batch_size, 4096, 512)
if cmd_args.ipex:
    import intel_pytorch_extension as ipex
    print("Running with IPEX...")
    if cmd_args.precision == "bfloat16":
        # Automatically mix precision
        ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        print('Running with bfloat16...')
    input = input.to(ipex.DEVICE)
elif cmd_args.with_cuda:
    input = input.cuda()

model.eval()
if cmd_args.ipex:
    model.to(ipex.DEVICE)
    # if cmd_args.jit:
    #     model = torch.jit.trace(model, input)
elif cmd_args.with_cuda:
    model.cuda()

# training

batch_time = AverageMeter()
with torch.no_grad():
    for i in range(cmd_args.max_iters + cmd_args.warmup):
        if i >= cmd_args.warmup:
            start = time.time()
        output = model(input)
        if i >= cmd_args.warmup:
            batch_time.update(time.time() - start)
        # if i % 10 == 0:
        print("iterations:{}/({})".format(i+1, cmd_args.max_iters + cmd_args.warmup))

    latency = batch_time.avg / cmd_args.batch_size * 1000
    perf = cmd_args.batch_size / batch_time.avg
    print('inference latency: %0.3f ms' % latency)
    print('inference Throughput: %0.3f samples/s' % perf)