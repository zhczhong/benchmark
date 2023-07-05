import os
import json
import argparse

from operator import itemgetter


class Profile:
    def __init__(self, name, time, calls, type):
        self.name = name
        self.time = time
        self.calls = calls
        self.type = type
    def __repr__(self):
        return repr((self.name, self.time, self.calls, self.type))


def Parse_File(file_path, display_op):
    # load json file
    data = ''
    try:
        # Reading data back
        with open(file_path, 'r') as f:
            data = json.load(f)['traceEvents']
    except Exception as e:
        print("Profiling json file parse ERROr! ")
    # filter and sort
    filter_data = []
    for detail_dict in data:
        if 'ts' in detail_dict.keys():
            filter_data.append(detail_dict)
    sorted_data = sorted(filter_data, key=itemgetter('ts'))
    # generate op detials
    op_list = []
    total_time = 0
    if data is not None:
        op_type = "FP32"
        tmp_type = op_type
        for detail_dict in sorted_data:
            if ('cat' in detail_dict.keys() and detail_dict['cat'] == 'Op') \
                or ('ph' in detail_dict.keys() and detail_dict['ph'] == 'X'):
                op_name = str(detail_dict['name'])
                op_time = int(detail_dict['dur']) / 1000
                op_calls = 1
                total_time += op_time
                if "args" in detail_dict:
                    for key, value in detail_dict['args'].items():
                        if op_name == "Cast" and 'name' in key.lower() and 'CastToBf16' in value:
                            op_type = "BF16"
                            tmp_type = op_type
                            break
                        elif 'input' in key.lower() and 'CastToBf16' in value:
                            op_type = "BF16"
                            tmp_type = op_type
                            break
                        elif op_name == "Cast" and 'name' in key.lower() and 'CastToFp32' in value:
                            op_type = "FP32"
                            tmp_type = op_type
                            break
                        elif 'input' in key.lower() and 'CastToFp32' in value:
                            op_type = "FP32"
                            tmp_type = op_type
                            break
                        elif 'quantize' in op_name.lower():
                            tmp_type = op_type
                            op_type = "INT8"
                            break
                        # elif "FP32toBF16" in value and 'input' in key.lower():
                        elif op_name == "Cast" and 'name' in key.lower() and 'FP32toBF16' in value:
                            op_type = "BF16"
                            tmp_type = op_type
                            break
                        # elif "BF16toFP32" in value and 'input' in key.lower():
                        elif op_name == "Cast" and 'name' in key.lower() and 'BF16toFP32' in value:
                            op_type = "FP32"
                            tmp_type = op_type
                            break

                if display_op is not None:
                    if op_name != display_op:
                        continue
                
                if op_name + op_type not in [op.name + op.type for op in op_list]:
                    op_list.append(Profile(op_name, op_time, op_calls, op_type))
                else:
                    for index, op in enumerate(op_list):
                        if op_name + op_type == op.name + op.type:
                            op_list[index].time += op_time
                            op_list[index].calls += op_calls
            op_type = tmp_type
    # sort
    op_list = sorted(op_list, key=lambda op: op.time, reverse=True)

    # print op
    print("Operator\tDuration(ms)\tOcurrences\tPercentage(%)\tDataType")
    for op in op_list:
        col1 = op.name
        col2 = op.time
        col3 = op.calls
        col4 = op.time / total_time * 100
        col5 = op.type
        print("{}\t{:.3f}\t{:.0f}\t{:.0f}    {}".format(col1, col2, col3, col4, col5))
    print("Total: %s\n\n" % (total_time))

    return op_list


if __name__ == "__main__":
    # parar args
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, default="", help="path of profiling json file")
    parser.add_argument("-p", "--display_op", type=str, default=None, help="op name to display")
    args = parser.parse_args()

    Parse_File(args.file_path, args.display_op)
