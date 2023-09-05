import os
import sys
import time
import logging
import argparse
import math
import yaml
import numpy as np
import torch

from contextlib import nullcontext
from tensorflow.python.client import timeline
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.protobuf import rewriter_config_pb2
from find_outputs import get_input_output
from utils import *

logging.basicConfig(level=logging.INFO,
                    datefmt='[%H:%M:%S]',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OOB-Benchmark")


def metrics_generator(array, tolerance):
    max_diff = np.max(array)
    mean_diff = np.mean(array)
    median_diff = np.median(array)
    success_rate = np.sum(array < tolerance) / array.size
    return max_diff, mean_diff, median_diff, success_rate

def initialize_graph(model_details, args, od_graph_def):
    graph = tf_v1.Graph()
    with graph.as_default():
        input_variables = {
            in_name + ":0": tf_v1.Variable(val)
            for in_name, val in model_details['input'].items()}

        if not args.use_nc:
            with tf_v1.gfile.GFile(os.path.join(os.getcwd(), model_details['model_dir']), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                od_graph_def = delete_assign(od_graph_def)

        elif args.use_nc and not od_graph_def.node:
            from neural_compressor.experimental import common
            model = common.Model(os.path.join(os.getcwd(), model_details['model_dir']))
            od_graph_def = model.graph_def

        # optimize for inference
        if not args.disable_optimize:
            # optimize graph for inference
            try:
                input_list = [ in_name for in_name,val in model_details['input'].items() ]
                output_list = [ out_name for out_name in model_details['output'] ]
                input_data_type = [ tf_v1.convert_to_tensor(item).dtype.as_datatype_enum for item in model_details['input'].values() ]

                od_graph_def_tmp = od_graph_def
                od_graph_def = optimize_for_inference_lib.optimize_for_inference(
                    od_graph_def,  # inputGraph,
                    input_list,  # an array of the input nodes
                    output_list,  # an array of output nodes
                    input_data_type)
                od_graph_def.library.CopyFrom(od_graph_def_tmp.library)
                print("---- Optimized for inference!")
            except:
                print("---- Optimize for inference failed!")

        tf_v1.import_graph_def(od_graph_def, name='g',
                            input_map=input_variables)

    return graph

def create_tf_config(args):
    if "OMP_NUM_THREADS" in os.environ:
        OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
    else:
        OMP_NUM_THREADS = len(os.sched_getaffinity(0))

    config = tf_v1.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = OMP_NUM_THREADS
    config.inter_op_parallelism_threads = 1
    # additional options
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
    config.graph_options.rewrite_options.function_optimization = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
    # LLGA add
    config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.cpu_layout_conversion = rewriter_config_pb2.RewriterConfig.NCHW_TO_NHWC
    if args.precision == 'bfloat16':
        # config.graph_options.rewrite_options.auto_mixed_precision_onednn_bfloat16 = rewriter_config_pb2.RewriterConfig.ON
        config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
        config.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON
    return config

def run_benchmark(model_details, args, find_graph_def):
    tf_config = create_tf_config(args)
    # disable onednn graph as correctness reference
    tf_config_ref = create_tf_config(args)
    # tf_config_ref.graph_options.onednn_graph = 0 
    graph = initialize_graph(model_details, args, find_graph_def)
    run_options = tf_v1.RunOptions(trace_level=tf_v1.RunOptions.FULL_TRACE)
    run_metadata = tf_v1.RunMetadata()

    if args.save_graph:
        # write the real benchmark graph to local
        model_dir = os.path.dirname(os.path.abspath(model_detail['model_dir']))
        out_graph_file = os.path.join(model_dir, 'runtime_graph.pb')
        write_graph(graph.as_graph_def(), out_graph_file)
        print("********** save runtime graph at {}".format(out_graph_file))

    print("========= start running", flush=True)
    with tf_v1.Session(config=tf_config, graph=graph) as sess, tf_v1.Session(config=tf_config_ref, graph=graph) if args.check_correctness else nullcontext() as sess_ref:
        output_dict = {out_name: graph.get_tensor_by_name("g/" + out_name + ":0")
                       for out_name in model_details['output']}

        sess.run(tf_v1.global_variables_initializer())
        if args.check_correctness:
            sess_ref.run(tf_v1.global_variables_initializer())

        total_time = 0.0
        reps_done = 0
        reps_correctness_passed = 0
        for rep in range(args.num_iter):
            # sess run
            start = time.time()
            run_result = []
            if args.profile:
                run_result = sess.run(output_dict, options=run_options, run_metadata=run_metadata)
            else:
                print("============================ run model with llga enabled", flush=True)
                run_result = sess.run(output_dict) 
            end = time.time()
            delta = end - start
            print("Iteration: {}, inference time: {} sec".format(rep, delta))
            
            if args.check_correctness:
                print("============================ run model with llga disabled as reference", flush=True)

                import intel_extension_for_tensorflow as itex
                graph_options = itex.GraphOptions()
                graph_options.onednn_graph = itex.OFF
                config = itex.ConfigProto(graph_options=graph_options)
                itex.set_config(config)
                
                ref_result = sess_ref.run(output_dict)
            
                graph_options.onednn_graph = itex.ON
                config = itex.ConfigProto(graph_options=graph_options)
                itex.set_config(config)
                 
                result = same(ref_result, run_result)
                if result:
                    print("Iteration: {} check correctness PASS".format(rep), flush=True)
                    reps_correctness_passed += 1
                else:
                    print("Iteration: {} check correctness FAIL".format(rep), flush=True)

            if rep >= args.num_warmup and rep < (args.num_iter - args.num_warmup):
                total_time += delta
                reps_done += 1

            # save profiling file
            if args.profile and rep == int(args.num_iter / 2):
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                # model_dir = os.path.dirname(os.path.abspath(model_detail['model_dir']))
                model_dir = str(os.path.dirname(os.path.realpath(__file__))) + '/timeline'
                if not os.path.exists(model_dir):
                    try:
                        os.makedirs(model_dir)
                    except:
                        pass
                profiling_file = model_dir + '/timeline-' + str(rep) + '-' + str(os.getpid()) + '.json'
                with open(profiling_file, 'w') as trace_file:
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))

        avg_time = total_time / reps_done
        latency = avg_time * 1000
        throughput = 1.0 / avg_time * args.batch_size
        print('Batch size = %d' % args.batch_size, flush=True)
        print("Latency: {:.3f} ms".format(latency))
        print("Throughput: {:.2f} fps".format(throughput))
        if args.check_correctness:
            print("CorrectnessCheck: {} out of {} iter passed".format(reps_correctness_passed, args.num_iter))

def same(ref, dst):
    print(ref)
    print(dst)
    ref_keys = list(ref.keys())
    dst_keys = list(dst.keys())
    ref_keys.sort()
    dst_keys.sort()
    print(ref_keys)
    print(dst_keys)
    if len(ref_keys) != len(dst_keys):
        return False
    for i in range(len(ref_keys)):
        if ref_keys[i] != dst_keys[i]:
            return False
    for key in ref_keys:
        ref_val = torch.from_numpy(ref.get(key))
        dst_val = torch.from_numpy(dst.get(key))
        ref_val = torch.flatten(ref_val)
        dst_val = torch.flatten(dst_val)
        if torch.equal(ref_val, dst_val):
            print("Output {} correctness check all equal".format(key))
            continue
        else:
            ref_val = ref_val.to(torch.float32)
            dst_val = dst_val.to(torch.float32)
            cos_sim = torch.nn.functional.cosine_similarity(ref_val, dst_val, dim=0, eps=1e-6)
            print("Output {} correctness check cos_sim: {}".format(key, cos_sim))
            if cos_sim < 0.99:
                return False
    return True

def _write_inputs_outputs_to_yaml(yaml_path, output_yaml_path, inputs, outputs):
    # deal with the inputs/outputs at yaml
    with open(yaml_path, 'r') as f:
        content = yaml.safe_load(f)

        tmp_i = ''
        tmp_o = ''
        for item in inputs:
            tmp_i = tmp_i + str(item) + ','
        for item in outputs:
            tmp_o = tmp_o + str(item) + ','
        content['model'].update({'inputs': tmp_i[:-1]})
        content['model'].update({'outputs': tmp_o[:-1]})
        print(content)

    with open(output_yaml_path, 'w') as nf:
        yaml.dump(content, nf)

def oob_collate_data_func(batch):
    """Puts each data field into a pd frame with outer dimension batch size"""
    elem = batch[0]
    import collections
    if isinstance(elem, collections.abc.Mapping):
        return {key: oob_collate_data_func([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [oob_collate_data_func(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        return np.stack(batch)
    elif isinstance(elem, bool) or isinstance(elem, np.bool_):
        return elem
    else:
        return batch

def oob_collate_sparse_func(batch):
    """Data collation function for sparse dummy dataset"""
    extract_batch = []
    for i in batch[0]:
        for j in i[0]:
            extract_batch.append(j)
    return tuple(extract_batch[idx] for idx in seq_idxs), 0

def oob_dlrm_collate_func(batch):
    """Data collation function for DLRM"""
    dense_features = np.array([[0., 1.3862944, 1.3862944, 1.609438, 8.13798, 5.480639, 0.6931472,
                               3.218876, 5.187386, 0., 0.6931472, 0., 3.6635616]], dtype=np.float32)
    sparse_features = np.array([[3, 93, 319, 272, 0, 5, 7898, 1, 0, 2, 3306, 310, 2528, 7,
                                293, 293, 1, 218, 1, 2, 302, 0, 1, 120, 1, 2]], dtype=np.int32)
    return (dense_features, sparse_features), 0

class DataLoader(object):
    def __init__(self, inputs_tensor, total_samples, batch_size):
        """dataloader generator

        Args:
            data_location (str): tf recorder local path
            batch_size (int): dataloader batch size
        """
        self.batch_size = batch_size
        self.inputs_tensor = inputs_tensor
        # self.input_dtypes = input_dtypes
        self.total_samples = total_samples
        self.n = math.ceil(float(self.total_samples) / self.batch_size)
        # assert len(input_shapes) == len(input_dtypes)
        print("batch size is " + str(self.batch_size) + "," + str(self.n) + " iteration")

    def __iter__(self):
        for _ in range(self.n):
            if len(self.inputs_tensor.values()) > 1:
                data = [list(self.inputs_tensor.values())]
            else:
                data = list(self.inputs_tensor.values())
            yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-m", "--model_name", help="name of model")
    parser.add_argument("-pb", "--model_path", help="path of model")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--precision", type=str, default='float32', help="float32, int8 or bfloat16")
    parser.add_argument("-i", "-n", "--num_iter", type=int, default=1, help="numbers of inference iteration, default is 500")
    parser.add_argument("-w", "--num_warmup", type=int, default=0, help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_true', help="use this to disable optimize_for_inference")
    parser.add_argument("--profile", action='store_true', help="profile")
    parser.add_argument("--is_meta", action='store_true', help="input a meta file")
    parser.add_argument("--save_graph", action='store_true', help="save_graph")
    parser.add_argument("--benchmark", action='store_true', help="Benchmark.")
    parser.add_argument("--use_nc", action='store_true', help="Find input/output via neural_compressor.")
    parser.add_argument("--output_name", nargs='*', help="Specify output for neural_compressor ckpt.")
    parser.add_argument("--check_correctness", action='store_true', help="to do correctness check")
    # tuning
    parser.add_argument("--yaml", type=str, help="config yaml file of neural_compressor.", default='./config.yaml')
    parser.add_argument("--tune", action='store_true', help="Do neural_compressor optimize.")
    parser.add_argument("--output_path", help="path of neural_compressor convert model", default='./nc-tune.pb')
    # args
    args = parser.parse_args()
    
    # benchmark PB model directly
    find_graph_def = tf_v1.GraphDef()
    if args.model_path and not args.model_name:
        # generate model detail
        model_dir = args.model_path
        model_detail = {}
        model_detail['a_row_max'] = []
        model_detail['a_column_max'] = []
        find_graph_def, model_input_output = get_input_output(model_dir, args)
        # ckpt/meta model will save freezed pb in the same dir
        model_dir = model_dir if not args.is_meta else args.model_path[:-5] + "_freeze.pb"
        output = model_input_output['outputs']
        input_dic = {}
        input_nodes_info = model_input_output['inputs']['input_nodes_info']
        for _input in input_nodes_info:
            # deal with bool dtype input
            if input_nodes_info[_input]['type'] == 'bool':
                input_dic[_input] = input_nodes_info[_input]['value']
            elif _input == 'dropout_keep_prob':
                input_dic[_input] = np.array([0.5,], dtype='float32')
            else:
                dtype = input_nodes_info[_input]['type']
                dshape = input_nodes_info[_input]['shape']
                is_one_dim = input_nodes_info[_input]['is_one_dim']
                sparse_d_shape_ops = model_input_output['inputs'].get('sparse_d_shape', {})
                sparse_d_shape_op = [i for i in sparse_d_shape_ops.values() if _input in i]
                if sparse_d_shape_op and list(sparse_d_shape_op[0]).index(_input)==0:
                    dense_shape = sparse_d_shape_op[0][_input]
                    dummy_input = generate_sparse_indice(dense_shape, dtype, args.batch_size)
                else:
                    dummy_input = generate_data(dshape, dtype, args.batch_size, is_one_dim=is_one_dim)
                input_dic[_input] = dummy_input
        model_detail['model_dir'] = model_dir
        model_detail['input'] = input_dic
        model_detail['output'] = output
        model_detail['ckpt'] = args.is_meta
        model_detail['sparse_d_shape'] = model_input_output['inputs'].get('sparse_d_shape', {})

    # benchmark with input/output
    elif args.model_name:
        # handle the case that the original input is deleted
        # if args.benchmark and args.model_name == 'deepspeech':
        #     args.model_name = 'deepspeech-tuned'
        assert args.model_path is not None, "Model path is undefined."
        from model_detail import models
        model_detail = None
        for model in models:
            if model['model_name'] == args.model_name:
                model_detail = model
                model_detail['model_dir'] = args.model_path
                model_detail['ckpt'] = args.is_meta
                break
        if not model_detail:
            logger.error("Model undefined.")
            sys.exit(1)
    
    inputs_shape = []
    inputs_dtype = []
    for input_tensor in model_detail['input'].values():
        if not isinstance(input_tensor, bool):
            inputs_shape.append(input_tensor.shape)
            inputs_dtype.append(str(input_tensor.dtype))
        else:
            # TODO: wait scalar support in dummy dataset
            inputs_shape.append((1,))
            inputs_dtype.append('bool')
    print("Final benchmark input nodes: name_list={}, shape_list={}, dtype_list={}".format( \
                list(model_detail['input'].keys()), inputs_shape, inputs_dtype))
    print("Final benchmark output nodes: name_list={}".format(model_detail['output']))

    # tune
    if args.tune:
        from dataloaders import WidedeepDataloader
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from neural_compressor.experimental import Quantization, common
        inputs = model_detail['input']
        outputs = model_detail['output']
        _write_inputs_outputs_to_yaml(args.yaml, "./config_tmp.yaml", list(inputs.keys()), outputs)

        quantizer = Quantization("./config_tmp.yaml")
        # generate dummy data
        if model_detail.get('sparse_d_shape'):
            sparse_input_names = [list(i.keys()) for i in model_detail['sparse_d_shape'].values()]
            sparse_input_seq = sparse_input_names[0]
            for i in range(1, len(sparse_input_names)):
                sparse_input_seq += sparse_input_names[i]
            input_dense_shape = [tuple(list(i.values())[0]) for i in model_detail['sparse_d_shape'].values()]
            dataset = quantizer.dataset(dataset_type='sparse_dummy_v2',
                                        dense_shape=input_dense_shape,
                                        label_shape=[[1] for _ in range(len(input_dense_shape))],
                                        sparse_ratio=[1-1/np.multiply(*i) for i in input_dense_shape])
            seq_idxs = [sparse_input_seq.index(i) for i in inputs.keys()]
            quantizer.calib_dataloader = common.DataLoader(dataset=dataset,
                                                           batch_size=1,
                                                           collate_fn=oob_collate_sparse_func)
        else:
            dataset = quantizer.dataset(dataset_type='dummy',
                                        shape=inputs_shape,
                                        low=1.0, high=20.0,
                                        dtype=inputs_dtype,
                                        label=True)
            dataloader_dict = {'wide_deep': WidedeepDataloader}
            if args.model_name and args.model_name in dataloader_dict.keys():
                Dataloader = dataloader_dict[args.model_name]
            else:
                Dataloader = common.DataLoader
            quantizer.calib_dataloader = Dataloader(dataset=dataset,
                                                    batch_size=args.batch_size,
                                                    collate_fn=oob_collate_data_func \
                                                        if model_detail.get('model_name')!='DLRM' \
                                                            else oob_dlrm_collate_func)
        quantizer.model = args.model_path
        q_model = quantizer.fit()
        q_model.save(args.output_path)

    # benchmark
    if args.benchmark:
        run_benchmark(model_detail, args, find_graph_def)

