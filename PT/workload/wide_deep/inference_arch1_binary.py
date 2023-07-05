import pandas as pd
import torch
import os
import time
import argparse
from tqdm import trange
from pytorch_widedeep.preprocessing import WidePreprocessor, DeepPreprocessor
from pytorch_widedeep.models import Wide, DeepDense, WideDeep
from pytorch_widedeep.metrics import BinaryAccuracy
from pytorch_widedeep.models._wd_dataset import WideDeepDataset

from torch.utils.data import DataLoader
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--num_warmup', type=int, default=5, help='input batch size')
parser.add_argument('--num_workers', type=int, default=1, help='input batch size')
parser.add_argument('--num_iter', type=int, default=0, help='input batch size')
parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--pretrained', default='./model/wide_deep.pth', help="path to pretrained model")
parser.add_argument('--save_dir', default='./model/', help='Where to store models')
parser.add_argument('--inf', action='store_true', help='inference only')
parser.add_argument('--ipex', action='store_true', default=False, help='Use ipex to get boost.')
# parser.add_argument('--jit', action='store_true', default=False,
#                      help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--precision', default='float32', help='precision, "float32" or "bfloat16"')
parser.add_argument('--profiling', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)

args = parser.parse_args()
print(args)


def dataProcess():
    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    # these next 3 lines are not directly related to pytorch-widedeep. I assume
    # you have downloaded the dataset and place it in a dir called data/adult/
    df = pd.read_csv('data/adult.data', names=_CSV_COLUMNS)

    df['income_label'] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop('income', axis=1, inplace=True)
    return df

def _get_wide_deep():
    df = dataProcess()
    # prepare wide, crossed, embedding and continuous columns
    wide_cols  = ['education', 'relationship', 'workclass', 'occupation','native-country', 'gender']
    cross_cols = [('education', 'occupation'), ('native-country', 'occupation')]
    embed_cols = [('education',16), ('workclass',16), ('occupation',16),('native-country',32)]
    cont_cols  = ["age", "hours-per-week"]
    target_col = 'income_label'

    # target
    target = df[target_col].values
    # wide
    preprocess_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
    X_wide = preprocess_wide.fit_transform(df)
    wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)

    # deepdense
    preprocess_deep = DeepPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
    X_deep = preprocess_deep.fit_transform(df)
    deepdense = DeepDense(hidden_layers=[64,32],
                        deep_column_idx=preprocess_deep.deep_column_idx,
                        embed_input=preprocess_deep.embeddings_input,
                        continuous_cols=cont_cols)


    return X_wide, wide, X_deep, deepdense, target
    
def train():

    X_wide, wide, X_deep, deepdense, target = _get_wide_deep()

    # build, compile, fit and predict
    model = WideDeep(wide=wide, deepdense=deepdense)
    model.compile(method='binary', metrics=[BinaryAccuracy])
    model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=args.nepoch, batch_size=args.batch_size, val_split=0.2)
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_model = save_path + "wide_deep.pth"
    torch.save(model, save_model)

def inference():
    X_wide, wide, X_deep, deepdense, _ = _get_wide_deep()
    load_dict = {"X_wide": X_wide, "X_deep": X_deep}
    test_set = WideDeepDataset(**load_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1
    model = torch.load(args.pretrained)
    if args.ipex:
        import intel_pytorch_extension as ipex
        print("Running with IPEX...")
        if args.precision == "bfloat16":
            # Automatically mix precision
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
            print("Running with bfloat16...")
        model = model.to(ipex.DEVICE)
    # if args.jit:
    #     # model = torch.jit.trace(model.eval(), X_wide, X_deep)
    #     model = torch.jit.script(model.eval())

    totle_time = 0
    with torch.no_grad():
        with trange(test_steps, disable=False) as t:
            for i, data in zip(t, test_loader):
                if i >= args.num_warmup:
                    tic = time.time()
                t.set_description("predict")
                if args.ipex:
                    X = {k: v.to(ipex.DEVICE) for k, v in data.items()}
                else:
                    X = data
                if args.profiling and i > 0:
                    with torch.autograd.profiler.profile(use_cuda=False) as prof:
                        preds = model._activation_fn(model.forward(X))
                        if model.method == "multiclass":
                            preds = F.softmax(preds, dim=1)
                        if i == args.num_warmup:
                            break
                else:
                    preds = model._activation_fn(model.forward(X))
                    if model.method == "multiclass":
                        preds = F.softmax(preds, dim=1)
                preds = preds.cpu().data.numpy()

                if i >= args.num_warmup:
                    totle_time += time.time() - tic

                if args.num_iter > 0 and i >= args.num_iter + args.num_warmup:
                    break

    if args.profiling:
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    else:
        throughput = (i - args.num_warmup + 1) * test_loader.batch_size / totle_time
        print("Latency: %s ms" % str(1 / throughput))
        print("Throughput: %s samples/sec" % str(throughput))

    return


def main():
    if args.eval:
        inference()
    else:
        train()


if __name__ == '__main__':
    main()


