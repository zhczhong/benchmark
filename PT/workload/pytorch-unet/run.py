import time
import numpy as np
import simulation
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pytorch_unet

parser = argparse.ArgumentParser(description='PyTorch UNet evaluation')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--n_classes', default=6, type=int,
                    metavar='N', help='the number of class')
parser.add_argument('-i', '--iterations', default=100, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w', '--warmup-iterations', default=10, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision, float32, int8, bfloat16')

args = parser.parse_args()

if args.ipex:
    import intel_pytorch_extension as ipex
    print("import IPEX **************")
    if args.precision == "bfloat16":
        # Automatically mix precision
        print("Running with bfloat16...")
        ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)        
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    print(args)

    # use same transform for train/val for this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_set = SimDataset(200, transform = trans)
    dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.ipex:
        device = ipex.DEVICE
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = pytorch_unet.UNet(args.n_classes).eval()
    model.to(device)
    if args.jit:
        print("Running with jit script model...")
        model = torch.jit.script(model)

    batch_time = AverageMeter()
    for i in range(args.iterations + args.warmup_iterations):
        inputs, labels = next(iter(dataloader))
        if i >= args.warmup_iterations:
            start = time.time()
        inputs = inputs.to(device)
        labels = labels.to(device)

        pred = model(inputs)

        if i >= args.warmup_iterations:
            batch_time.update(time.time() - start)

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i, args.iterations + args.warmup_iterations, batch_time=batch_time))

    latency = batch_time.avg / args.batch_size * 1000
    perf = args.batch_size/batch_time.avg
    print('inference latency: %3.3f ms'%latency)
    print('inference Throughput: %3.3f fps'%perf)


if __name__ == '__main__':
    main()