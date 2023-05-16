from vit_pytorch.distill import DistillableViT, DistillWrapper
import vit_pytorch
from onnx2pytorch import ConvertModel
import argparse
import os
import random
import time
import PIL
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

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


def get_vit_model_list():
    from vit_pytorch.deepvit import DeepViT
    from torchvision.models import resnet50
    from vit_pytorch.cait import CaiT
    from vit_pytorch.t2t import T2TViT
    from vit_pytorch.cct import CCT
    vit_model_mapping = {
        "DeepViT":
        DeepViT(image_size=256,
                patch_size=32,
                num_classes=1000,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1),
        "DistillableViT":
        DistillWrapper(student=DistillableViT(image_size=256,
                                              patch_size=32,
                                              num_classes=1000,
                                              dim=1024,
                                              depth=6,
                                              heads=8,
                                              mlp_dim=2048,
                                              dropout=0.1,
                                              emb_dropout=0.1),
                       teacher=resnet50(pretrained=True),
                       temperature=3,
                       alpha=0.5,
                       hard=False),
        "CaiT":
        CaiT(
            image_size=256,
            patch_size=32,
            num_classes=1000,
            dim=1024,
            depth=12,  # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05  # randomly dropout 5% of the layers
        ),
        "T2TVIT":
        T2TViT(
            dim=512,
            image_size=224,
            depth=5,
            heads=8,
            mlp_dim=512,
            num_classes=1000,
            # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
            t2t_layers=((7, 4), (3, 2), (3, 2))),
        "CCT":
        CCT(
            img_size=(224, 448),
            embedding_dim=384,
            n_conv_layers=2,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_layers=14,
            num_heads=6,
            mlp_ratio=3.,
            num_classes=1000,
            positional_embedding='learnable',  # ['sine', 'learnable', 'none']
        )
    }
    return vit_model_mapping


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    metavar='DIR',
                    default='',
                    help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--model-source',
                    metavar='MODEL_SOURCE',
                    default='torchvision',
                    help='model source(timm, torchvision, geffnet)')
parser.add_argument('-j',
                    '--workers',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--image_size', default=224, type=int, help='image size')
parser.add_argument('--advprop',
                    default=False,
                    action='store_true',
                    help='use advprop or not')
parser.add_argument('--ipex',
                    action='store_true',
                    default=False,
                    help='use ipex weight cache')
parser.add_argument('--jit',
                    action='store_true',
                    default=False,
                    help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--llga',
                    action='store_true',
                    default=False,
                    help='enable LLGA')
parser.add_argument('--precision',
                    type=str,
                    default="float32",
                    choices=["f32", "fp32", "float32", "int8", "bfloat16"],
                    help='precision, float32, int8, bfloat16')
parser.add_argument('-i',
                    '--iterations',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w',
                    '--warmup-iterations',
                    default=10,
                    type=int,
                    metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument("-t",
                    "--profile",
                    action='store_true',
                    help="Trigger profile on current topology.")
parser.add_argument("--performance",
                    action='store_true',
                    help="measure performance only, no accuracy.")
parser.add_argument(
    "--dummy",
    action='store_true',
    help="using  dummu data to test the performance of inference")
parser.add_argument('--num-classes',
                    type=int,
                    default=1000,
                    help='Number classes in dataset')
parser.add_argument('--channels_last',
                    type=int,
                    default=1,
                    help='use channels last format')
parser.add_argument('--config_file',
                    type=str,
                    default="./conf.yaml",
                    help='config file for int8 tune')
parser.add_argument(
    "--weight-sharing",
    action='store_true',
    default=False,
    help="using weight_sharing to test the performance of inference")
parser.add_argument(
    "--number-instance",
    default=0,
    type=int,
    help=
    "the instance numbers for test the performance of latcy, only works when enable weight-sharing"
)
parser.add_argument("--check_correctness",
                    action='store_true',
                    help="check correctness.")

args = parser.parse_args()
MAIN_RANDOM_SEED = 1
model_eager = None

if args.ipex:
    import intel_extension_for_pytorch as ipex
    print("Running with IPEX...")
    if args.precision == "bfloat16":
        # Automatically mix precision
        ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        print("Running with bfloat16...")


def main():
    global model_eager
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model, example_input = get_model(args.arch, args)
    model_eager = model

    criterion = nn.CrossEntropyLoss()

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

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
        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
            valdir, val_transforms),
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)

    if args.weight_sharing:
        assert args.dummy and args.batch_size, \
            "please using dummy data and set batch_size >= 1 if you want run weight sharing case for latency case"

    if args.evaluate:
        if args.jit:
            try:
                model = torch.jit.script(model)
                # input_var = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
                # model = torch.jit.trace(model, input_var)
                print("---- With JIT enabled.")
            except:
                print("---- With JIT disabled.")

        if args.precision == "int8":
            import intel_extension_for_pytorch as ipex
            from intel_extension_for_pytorch.quantization import prepare, convert
            qconfig = ipex.quantization.default_static_qconfig
            input_var = example_input
            if example_input is None:
                input_var = torch.randn(args.batch_size, 3, args.image_size,
                                        args.image_size)
            prepared_model = prepare(model,
                                     qconfig,
                                     example_inputs=input_var,
                                     inplace=False)
            for batch_idx, (d, target) in enumerate(val_loader):
                print(
                    f'calibrated on batch {batch_idx} out of {len(val_loader)}'
                )
                prepared_model(d)
            converted_model = convert(prepared_model)
            model = converted_model
            model_eager = model

        validate(val_loader, model, criterion, args, example_input)
        return


def get_model(model_name, args):
    # create model
    example_input = None
    if args.model_source == "geffnet":
        import geffnet
        if args.jit:
            geffnet.config.set_scriptable(True)
        if 'efficientnet' in model_name:
            model = geffnet.create_model(model_name,
                                         num_classes=args.num_classes,
                                         in_chans=3,
                                         pretrained=args.pretrained)
        elif 'mixnet' in model_name or 'fbnetc_100' in model_name or 'spnasnet_100' in model_name:
            geffnet.create_model(model_name,
                                 num_classes=args.num_classes,
                                 in_chans=3,
                                 pretrained=args.pretrained)
    elif args.model_source == "torchvision":
        if model_name == "inception_v3":
            model = torchvision.models.get_model(model_name,
                                                 pretrained=args.pretrained,
                                                 aux_logits=True,
                                                 transform_input=False)
        elif model_name == "googlenet":
            model = torchvision.models.get_model(model_name,
                                                 pretrained=args.pretrained,
                                                 transform_input=False)
        else:
            model = torchvision.models.get_model(model_name,
                                                 pretrained=args.pretrained)
    elif args.model_source == "timm":
        import timm
        model = timm.create_model(model_name,
                                  pretrained=args.pretrained,
                                  scriptable=args.jit)
    elif args.model_source == "VIT":
        vit_model_mapping = get_vit_model_list()
        model = vit_model_mapping[model_name]
    elif args.model_source == "local":
        model = torch.load(model_name)
    elif args.model_source == "torchbench":
        from torchbenchmark import load_model_by_name
        m = load_model_by_name(model_name)(test="eval",
                                           jit=False,
                                           device="cpu",
                                           batch_size=args.batch_size)
        model = m.model
        try:
            example_input = m.example_inputs[0]
        except:
            pass
    elif args.model_source == "onnx":
        import onnx
        from onnx2torch import convert
        model = convert(model_name)
    else:
        model = torch.hub.load(args.model_source,
                               model_name,
                               pretrained=args.pretrained)
    model = model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model.eval(), example_input


def get_image_size(model_name):
    if model_name in params_dict:
        _, _, res, _ = params_dict[model_name]
    else:
        res = args.image_size
    return res


def run_weights_sharing_model(m,
                              tid,
                              args,
                              batch_time,
                              progress,
                              example_input=None):
    steps = args.iterations + args.warmup_iterations
    steps = steps if steps > 0 else 300
    start_time = time.time()
    num_images = 0
    time_consume = 0
    timeBuff = []
    x = example_input
    if example_input is None:
        x = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    if args.precision == "bfloat16":
        x = x.to(torch.bfloat16)
    if args.ipex or args.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        while num_images < steps:
            start_time = time.time()
            if not args.jit and args.precision == "bfloat16":
                with torch.cpu.amp.autocast(enabled=True,
                                            dtype=torch.bfloat16):
                    y = m(x)
            else:
                y = m(x)

            end_time = time.time()

            if num_images > args.warmup_iterations:
                elasped_time_this_round = end_time - start_time
                time_consume += elasped_time_this_round
                timeBuff.append(elasped_time_this_round)
                if tid == 1:
                    batch_time.update(elasped_time_this_round)
                    if num_images % args.print_freq == 0:
                        progress.print(num_images)
            num_images += 1
        fps = args.batch_size / batch_time.avg
        avg_time = time_consume * 1000 / (steps - args.warmup_iterations)
        timeBuff = np.asarray(timeBuff)
        p99 = np.percentile(timeBuff, 99)
        print('P99 Latency {:.2f} ms'.format(p99 * 1000))
        print(
            'Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %
            (tid, avg_time, fps))


def set_random_seed():
    import torch
    import random
    import numpy
    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)


# copied from https://github.com/pytorch/torchdynamo/blob/main/torchdynamo/utils.py#L411


def same(a, b, cos_similarity=False, atol=1e-4, rtol=1e-4, equal_nan=False):
    """Check correctness to see if a and b match"""
    import torch
    import math
    if isinstance(a, (list, tuple, torch.nn.ParameterList, torch.Size)):
        assert isinstance(b,
                          (list, tuple)), f"type mismatch {type(a)} {type(b)}"
        return len(a) == len(b) and all(
            same(ai, bi, cos_similarity, atol, rtol, equal_nan)
            for ai, bi in zip(a, b))
    elif isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(
            b.keys()), f"keys mismatch {set(a.keys())} == {set(b.keys())}"
        for k in a.keys():
            if not (same(
                    a[k], b[k], cos_similarity, atol, rtol,
                    equal_nan=equal_nan)):
                print("Accuracy failed for key name", k)
                return False
        return True
    elif isinstance(a, torch.Tensor):
        if a.is_sparse:
            assert b.is_sparse
            a = a.to_dense()
            b = b.to_dense()
        if not isinstance(b, torch.Tensor):
            return False
        if cos_similarity:
            # TRT will bring error loss larger than current threshold. Use cosine similarity as replacement
            a = a.flatten().to(torch.float32)
            b = b.flatten().to(torch.float32)
            res = torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-6)
            if res < 0.99:
                print(f"Similarity score={res.cpu().detach().item()}")
            return res >= 0.99
        else:
            return torch.allclose(a,
                                  b,
                                  atol=atol,
                                  rtol=rtol,
                                  equal_nan=equal_nan)
    elif isinstance(a, (str, int, type(None), bool, torch.device)):
        return a == b
    elif isinstance(a, float):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    elif is_numpy_int_type(a) or is_numpy_float_type(a):
        return (type(a) is type(b)) and (a == b)
    elif is_numpy_ndarray(a):
        return (type(a)
                is type(b)) and same(torch.from_numpy(a), torch.from_numpy(b),
                                     cos_similarity, atol, rtol, equal_nan)
    elif type(a).__name__ in (
            "MaskedLMOutput",
            "Seq2SeqLMOutput",
            "CausalLMOutputWithCrossAttentions",
            "LongformerMaskedLMOutput",
            "Instances",
            "SquashedNormal",
            "Boxes",
            "Normal",
            "TanhTransform",
            "Foo",
            "Variable",
    ):
        assert type(a) is type(b)
        return all(
            same(getattr(a, key), getattr(b, key), cos_similarity, atol, rtol,
                 equal_nan) for key in a.__dict__.keys())
    else:
        raise RuntimeError(f"unsupported type: {type(a).__name__}")


def correctness_check(model,
                      model_eager,
                      example_input,
                      cos_sim=True,
                      rounds=10,
                      atol=1e-4,
                      rtol=1e-4) -> bool:
    set_random_seed()
    for _i in range(rounds):
        x = torch.rand_like(example_input)
        eager_output = model_eager(x)
        cur_result = model(x)
        if not same(eager_output,
                    cur_result,
                    cos_similarity=cos_sim,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True):
            return False
    return True


def validate(val_loader, model, criterion, args, example_input):
    global model_eager
    iterations = args.iterations
    warmup = args.warmup_iterations
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(iterations + warmup,
                             batch_time,
                             losses,
                             top1,
                             top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        if args.llga:
            import intel_extension_for_pytorch as ipex
            if example_input is None:
                example_input = torch.rand(args.batch_size, 3, args.image_size,
                                           args.image_size)
            if args.precision == "bfloat16":
                import torch.fx.experimental.optimization as optimization
                with torch.cpu.amp.autocast(cache_enabled=False):
                    model = model.eval()
                    model_eager = model.eval()
                    try:
                        model = torch.jit.trace(model, example_input)
                    except:
                        model = torch.jit.script(model)
            else:
                try:
                    model = torch.jit.trace(model, example_input)
                except:
                    model = torch.jit.script(model)
            model = torch.jit.freeze(model)
            print("---- Enable LLGA.")

        print("Running with", args.precision)

        if args.dummy:
            if args.weight_sharing:
                threads = []
                for i in range(1, args.number_instance + 1):
                    thread = threading.Thread(target=run_weights_sharing_model,
                                              args=(model, i, args, batch_time,
                                                    progress, example_input))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
            else:
                images = example_input
                if example_input is None:
                    images = torch.randn(args.batch_size, 3, args.image_size,
                                         args.image_size)
                target = torch.arange(1, args.batch_size + 1).long()
                # print("Start convert to onnx!")
                # torch.onnx.export(model.module, images, args.arch + ".onnx", verbose=False)
                # print("End convert to onnx!")

                model_oob, input_oob = model, images
                if args.channels_last:
                    model_oob, input_oob = model, images
                    model_oob = model_oob.to(memory_format=torch.channels_last)
                    input_oob = input_oob.to(memory_format=torch.channels_last)
                model, images = model_oob, input_oob
                print(images.shape)
                print(images.stride())
                for i in range(iterations + warmup):
                    if i >= warmup:
                        end = time.time()

                    # compute output
                    if args.profile:
                        with torch.profiler.profile(activities=[
                                torch.profiler.ProfilerActivity.CPU
                        ]) as prof:
                            if args.precision == "bfloat16":
                                with torch.cpu.amp.autocast(
                                        enabled=True, dtype=torch.bfloat16):
                                    output = model(images)
                            else:
                                output = model(images)
                    else:
                        if args.precision == "bfloat16":
                            with torch.cpu.amp.autocast(enabled=True,
                                                        dtype=torch.bfloat16):
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

                    if args.profile:
                        with torch.profiler.profile(activities=[
                                torch.profiler.ProfilerActivity.CPU
                        ]) as prof:
                            if args.precision == "bfloat16":
                                with torch.cpu.amp.autocast(
                                        enabled=True, dtype=torch.bfloat16):
                                    output = model(images)
                            else:
                                output = model(images)
                    else:
                        if args.precision == "bfloat16":
                            with torch.cpu.amp.autocast(enabled=True,
                                                        dtype=torch.bfloat16):
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
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

        if args.profile:
            import pathlib
            timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
            if not os.path.exists(timeline_dir):
                os.makedirs(timeline_dir)
            timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                        args.arch + str(i + 1) + '-' + str(os.getpid()) + '.json'

            prof.export_chrome_trace(timeline_file)
            # table_res = prof.key_averages().table(sort_by="cpu_time_total")
            # save_profile_result(torch.backends.quantized.engine + "_result_average.xlsx", table_res)

        # TODO: this should also be done with the ProgressMeter
        if args.evaluate:
            batch_size = args.batch_size
            latency = batch_time.avg / batch_size * 1000
            perf = batch_size / batch_time.avg
            print('inference latency: %3.3f ms' % latency)
            print('inference throughput on master instance: %3.3f fps' % perf)

        if args.check_correctness:
            if args.channels_last:
                example_input = example_input.to(
                    memory_format=torch.channels_last)
            atol = 1e-3
            rtol = 1e-3
            result = correctness_check(model,
                                       model_eager,
                                       example_input,
                                       atol=atol,
                                       rtol=rtol)
            if result:
                print("Correctness result: Pass")
            else:
                print("Correctness result: Fail")

    return top1.avg


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


def accuracy(output, target, topk=(1, )):
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
    for i in range(3, len(lines) - 4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i - 2, j, word)
                j += 1
    workbook.close()


if __name__ == '__main__':
    main()
