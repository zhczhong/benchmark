import argparse
import os
import random
import shutil
import time
import warnings
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.fx.experimental.optimization as optimization

import pretrainedmodels.utils

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

params_dict = {
    # Coefficients:   width,depth,res,dropout
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
    'efficientnet_b8': (2.2, 3.6, 672, 0.5),
    'efficientnet_l2': (4.3, 5.3, 800, 0.5),
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--ppn', default=1, type=int,
                    help='number of processes on each node of distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex weight cache')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable JIT path')
parser.add_argument('--jit_optimize', action='store_true', default=False,
                    help='enable JIT-optimize-for-inference path')
parser.add_argument('--llga', action='store_true', default=False, help='enable LLGA')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision: float32, bfloat16, int8_ipex, int8_imperative, int8_fx')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('-i', '--iterations', default=200, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w', '--warmup-iterations', default=10, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument("-t", "--profile", action='store_true',
                    help="Trigger profile on current topology.")
parser.add_argument("--performance", action='store_true',
                    help="measure performance only, no accuracy.")
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--channels_last', type=int, default=1,
                    help='use channels last format')
parser.add_argument('--to_mkldnn', type=int, default=0,
                    help='use mkldnn')
parser.add_argument('--config_file', type=str, default="./conf.yaml",
                    help='config file for int8 tune')
parser.add_argument("--int8_mkldnn", action='store_true',
                    help="using int8_mkldnn engine")
parser.add_argument("--torchdynamo_ipex", action='store_true',
                    help="using torchdynamo with ipex backend")
parser.add_argument("--fx", action='store_true',
                    help="using fx.optimization.fuse")
parser.add_argument("--torchdynamo_fx", action='store_true',
                    help="using torchdynamo with fx backend")

args = parser.parse_args()

if args.ipex:
    import intel_extension_for_pytorch as ipex
    print("Running with IPEX...")

def main():
    args = parser.parse_args()
    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print("Using CUDA ...")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.cuda:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.ppn > 1 or args.multiprocessing_distributed

    if args.gpu is not None and args.cuda:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = args.ppn
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        import geffnet
        geffnet.config.set_scriptable(False) # this is to disable TE fusion brought by @torch.jit.script decorators in geffnet model definition
        if args.jit:
            geffnet.config.set_scriptable(True)
        if args.precision == "int8_ipex":
            geffnet.config.set_scriptable(True)
        if args.pretrained:
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=True)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=False)
        model.train(False)
    elif 'mixnet' in args.arch or 'fbnetc_100' in args.arch or 'spnasnet_100' in args.arch:
        import geffnet
        if args.jit:
            geffnet.config.set_scriptable(True)
        if args.pretrained:
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=True)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=False)
        model.train(False)
    elif 'nasnetalarge' in args.arch or 'dpn' in args.arch or 'vggm' in args.arch or 'inceptionresnetv2' in args.arch or 'polynet' in args.arch or 'se_resne' in args.arch or 'senet' in args.arch[0:4]:
        import pretrainedmodels
        import pretrainedmodels.utils
        if args.pretrained:
            model = pretrainedmodels.__dict__[args.arch](num_classes=args.num_classes, pretrained='imagenet')
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = pretrainedmodels.__dict__[args.arch](pretrained=None)
        model.train(False)
    elif 'fpn' in args.arch:
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.train(False)
        args.iterations = 20
        print("Will run only ", args.iterations, " iterations for this model.")
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            if args.arch == "inception_v3":
                model = models.__dict__[args.arch](pretrained=True, aux_logits=False, transform_input=False)
            else:
                if args.arch == "googlenet":
                    model = models.__dict__[args.arch](pretrained=True, transform_input=False)
                else:
                    model = models.__dict__[args.arch](pretrained=True)
        else:
            if args.arch == "inception_v3":
                print("=> creating model '{}'".format(args.arch))
                model = models.__dict__[args.arch](aux_logits=False)
            else:
                print("=> creating model '{}'".format(args.arch))
                model = models.__dict__[args.arch]()
        model.train(False)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None and args.cuda:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            if args.cuda:
                model.cuda()
                print("create DistributedDataParallel in GPU")
            else:
                print("create DistributedDataParallel in CPU")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        pass
        # # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     if args.cuda:
        #         model.cuda()
        # else:
        #     if not args.jit:
        #         model = torch.nn.DataParallel(model)
        #     if args.cuda:
        #         model.cuda()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.fx:
        model = optimization.fuse(model, inplace=True)
    if args.to_mkldnn and args.evaluate:
        model = torch.utils.mkldnn.to_mkldnn(model)
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        if args.precision in ["bfloat16", "bfloat16_brutal"]:
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        elif args.precision == "float32":
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        print("Running with IPEX {}...".format(args.precision))

    # define loss function (criterion) and optimizer
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss()
    if not args.evaluate:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and args.cuda:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if not args.dummy and args.data:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if 'efficientnet' in args.arch:
        image_size = get_image_size(args.arch)
        val_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        args.image_size = image_size
        print('Using image size', image_size)
    elif 'mixnet' in args.arch:
        image_size = 112
        args.image_size = image_size
        print('Using image size', image_size)
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', args.image_size)

    val_loader = []
    if not args.dummy and args.data:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, val_transforms),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        if args.jit and not args.jit_optimize:
            if args.channels_last:
                image = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous(memory_format=torch.channels_last)
            else:
                image = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous()
            if args.to_mkldnn:
                image = image.to_mkldnn()
            if args.cuda:
                image = image.cuda(args.gpu, non_blocking=True)
            if args.precision in ["bfloat16", "bfloat16_brutal"] and not args.cuda:
                if args.precision =="bfloat16_brutal":
                    image = image.to(torch.bfloat16)
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
                    print("Using CPU autocast to JIT ...")
                    if args.precision =="bfloat16_brutal":
                        model = model.to(dtype=torch.bfloat16)
                    model = torch.jit.trace(model, image, check_trace=False)
            elif args.precision in ["bfloat16", "bfloat16_brutal"] and args.cuda:
                if args.precision =="float16_brutal":
                    image = image.to(torch.float16)
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16), torch.no_grad():
                    print("Using CUDA autocast to JIT ...")
                    if args.precision =="float16_brutal":
                        model = model.to(dtype=torch.float16)
                    model = torch.jit.trace(model, image, check_trace=False)
            else:
                with torch.no_grad():
                    model = torch.jit.trace(model, image, check_trace=False)
            model = torch.jit.freeze(model)
            print("---- With JIT enabled.")
        if args.jit and args.jit_optimize:
            if args.channels_last:
                image = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous(memory_format=torch.channels_last)
            else:
                image = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous()
            if args.to_mkldnn:
                image = image.to_mkldnn()
            if args.cuda:
                image = image.cuda(args.gpu, non_blocking=True)
            if args.precision in ["bfloat16", "bfloat16_brutal"] and not args.cuda:
                if args.precision =="bfloat16_brutal":
                    image = image.to(torch.bfloat16)
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
                    print("Using CPU autocast to JIT.optimize ...")
                    if args.precision =="bfloat16_brutal":
                        model = model.to(dtype=torch.bfloat16)
                    model = torch.jit.optimize_for_inference(torch.jit.trace(model, image, check_trace=False))
            elif args.precision in ["bfloat16", "bfloat16_brutal"] and args.cuda:
                image = image.to(torch.float16)
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16), torch.no_grad():
                    print("Using CUDA autocast to JIT.optimize ...")
                    model = model.to(dtype=torch.float16)
                    model = torch.jit.optimize_for_inference(torch.jit.trace(model, image, check_trace=False))
            else:
                with torch.no_grad():
                    model = torch.jit.optimize_for_inference(torch.jit.trace(model, image, check_trace=False))
            print("---- With JIT.optimize_for_inference enabled.")
        if args.precision == "int8_ipex":
            import intel_extension_for_pytorch as ipex
            model = optimization.fuse(model)
            print("Running IPEX INT8 calibration step ...\n")
            conf = ipex.quantization.QuantConf(qscheme=torch.per_tensor_symmetric)
            with torch.no_grad():
                for i in range(10):
                    with ipex.quantization.calibrate(conf):
                        # compute output
                        if args.channels_last:
                            images = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous(memory_format=torch.channels_last)
                        else:
                            images = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous()
                        if args.to_mkldnn:
                            images = images.to_mkldnn()
                        print(".........Cooking config_for_ipex_int8.json..........")
                        output = model(images)
                conf.save("./config_for_ipex_int8.json")
                print(".........calibration step done..........")
            model = optimization.fuse(model, inplace=True)
            conf = ipex.quantization.QuantConf("./config_for_ipex_int8.json")
            if args.channels_last:
                x = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous(memory_format=torch.channels_last)
            else:
                x = torch.randn(args.batch_size, 3, args.image_size, args.image_size).contiguous()
            if args.to_mkldnn:
                x = x.to_mkldnn()
            model = ipex.quantization.convert(model, conf, x)
            print("Running IPEX INT8 evaluation step ...\n")
            
        if args.precision in ["bfloat16", "bfloat16_brutal"] and not args.cuda:
            print("Using CPU autocast ...")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                res = validate(val_loader, model, criterion, args)
        elif args.precision in ["bfloat16", "bfloat16_brutal"] and args.cuda:
            print("Using CUDA autocast ...")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                res = validate(val_loader, model, criterion, args)
        else:  
            res = validate(val_loader, model, criterion, args)
        
        # with open('res.txt', 'w') as f:
        #     print(res, file=f)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if not args.performance:
            if args.precision in ["bfloat16", "bfloat16_brutal"] and not args.cuda:
                print("Using CPU autocast ...")
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    acc1 = validate(val_loader, model, criterion, args)
            elif args.precision in ["bfloat16", "bfloat16_brutal"] and args.cuda:
                print("Using CUDA autocast ...")
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    acc1 = validate(val_loader, model, criterion, args)
            else:
                acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)


def get_image_size(model_name):
    if model_name in params_dict:
        _, _, res, _ = params_dict[model_name]
    else:
        assert False, "Unsupported model:{}".format(model_name)
    return res


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.iterations > 0 and i >= (args.warmup_iterations + args.iterations):
            break
        # measure data loading time
        if i >= args.warmup_iterations:
            data_time.update(time.time() - end)

        if args.gpu is not None and args.cuda:
            images = images.cuda(args.gpu, non_blocking=True)
        if args.cuda:
            target = target.cuda(args.gpu, non_blocking=True)
        elif args.channels_last:
            images = images.to(memory_format=torch.channels_last)
        if args.to_mkldnn:
            images = images.to_mkldnn()
            
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i >= args.warmup_iterations:
            batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    if args.performance:
        batch_size = train_loader.batch_size
        latency = batch_time.avg / batch_size * 1000
        perf = batch_size/batch_time.avg
        print('training latency: %3.0f ms on %d epoch'%(latency, epoch))
        print('training Throughput: %3.0f fps on %d epoch'%(perf, epoch))


def validate(val_loader, model, criterion, args):
    iterations = args.iterations
    warmup = args.warmup_iterations
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        if args.llga:
            torch._C._jit_set_llga_enabled(True)
            model = torch.jit.trace(model, torch.rand(args.batch_size, 3, args.image_size, args.image_size))
            model = torch.jit.freeze(model)
            print("---- Enable LLGA.")

        if args.precision in ["bfloat16", "bfloat16_brutal"]:
            # with torch.amp.autocast(enabled=True, configure=torch.bfloat16, torch.no_grad(): 
            print("Running with bfloat16...")
        if args.dummy:
            images = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
            target = torch.arange(1, args.batch_size + 1).long()
            if args.precision in ["bfloat16", "bfloat16_brutal"]:
                if args.precision =="bfloat16_brutal":
                    images = images.to(torch.bfloat16)
                    target = target.to(torch.bfloat16)
            # print("Start convert to onnx!")
            # torch.onnx.export(model.module, images, args.arch + ".onnx", verbose=False)
            # print("End convert to onnx!")
            ### to_oob (ch_last or to_mkldnn)
            model_oob, input_oob = model, images
            if args.channels_last:
                model_oob, input_oob = model, images
                #model_oob = model_oob.to(memory_format=torch.channels_last)
                input_oob = input_oob.to(memory_format=torch.channels_last)
            if args.to_mkldnn:
                input_oob = input_oob.to_mkldnn()
            if args.to_mkldnn and args.evaluate:
                model_oob = torch.utils.mkldnn.to_mkldnn(model_oob)
            model, images = model_oob, input_oob
            for i in range(iterations + warmup):
                if i >= warmup:
                    end = time.time()
                if args.gpu is not None and args.cuda:
                    images = images.cuda(args.gpu, non_blocking=True)
                if args.cuda:
                    target = target.cuda(args.gpu, non_blocking=True)
                # compute output

                if args.torchdynamo_ipex:
                    print("Running TorchDynamo with IPEX backend")
                    import torchdynamo
                    from torchdynamo.optimizations import backends
                    if args.precision == "float32":
                        with torchdynamo.optimize(backends.ipex_fp32), torch.no_grad():
                            if args.profile:
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                                    output = model(images)
                            else:
                                output = model(images)
                    if args.precision in ["bfloat16", "bfloat16_brutal"]:
                        with torchdynamo.optimize(backends.ipex_bf16), torch.no_grad(), torch.cpu.amp.autocast():
                            if args.profile:
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                                    output = model(images)
                            else:
                                output = model(images)
                elif args.torchdynamo_fx:
                    print("Running TorchDynamo with FX (torch.fx.GraphModule) backend")
                    from typing import List
                    import torchdynamo
                    def fx_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
                        return gm.forward  # return a python callable
                    if args.precision == "float32":
                        with torchdynamo.optimize(fx_compiler), torch.no_grad():
                            if args.profile:
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                                    output = model(images)
                            else:
                                output = model(images)
                    if args.precision == "bfloat16":
                        with torchdynamo.optimize(fx_compiler), torch.no_grad(), torch.cpu.amp.autocast():
                            if args.profile:
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                                    output = model(images)
                            else:
                                output = model(images)
                else:
                    if args.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                            output = model(images)
                    else:
                        output = model(images)
                  
                # measure elapsed time
                if i >= warmup:
                    batch_time.update(time.time() - end)

                if i % args.print_freq == 0:
                    progress.print(i)
        else:
            for i, (images, target) in enumerate(val_loader):
                if not args.evaluate or iterations == 0 or i < iterations + warmup:
                    if i >= warmup:
                        end = time.time()
                    if args.gpu is not None and args.cuda:
                        images = images.cuda(args.gpu, non_blocking=True)
                    if args.cuda:
                        target = target.cuda(args.gpu, non_blocking=True)
                    if args.precision in ["bfloat16", "bfloat16_brutal"]:
                        if args.precision =="bfloat16_brutal":
                            images = images.to(torch.bfloat16)
                            target = target.to(torch.bfloat16)
                    if args.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                            output = model(images)
                    else:
                        output = model(images)

                    # measure elapsed time
                    if i >= warmup:
                        batch_time.update(time.time() - end)

                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    if i % args.print_freq == 0:
                        progress.print(i)
                elif i == iterations + warmup:
                    break
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        if args.profile:
            import pathlib
            timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
            if not os.path.exists(timeline_dir):
                os.makedirs(timeline_dir)
            timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                        args.arch + str(i + 1) + '-' + str(os.getpid()) + '.json'

            prof.export_chrome_trace(timeline_file)
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
            # table_res = prof.key_averages().table(sort_by="cpu_time_total")
            # save_profile_result(torch.backends.quantized.engine + "_result_average.xlsx", table_res)

        # TODO: this should also be done with the ProgressMeter
        if args.evaluate:
            batch_size = args.batch_size
            latency = batch_time.avg / batch_size * 1000
            perf = batch_size/batch_time.avg
            print('inference latency: %3.3f ms'%latency)
            print('inference Throughput: %3.3f fps'%perf)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()
if __name__ == '__main__':
    main()
