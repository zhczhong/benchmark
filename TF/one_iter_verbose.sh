#!/bin/bash

# input & output
input_dir="$1"
if [ "${input_dir}" == "" ];then
    echo "Please set your input files: $0 /path/to/your/file_or_dir"
    exit 1
fi

#
for var in `find $input_dir -type f -name "*.log"`
do
    # clearify log
    echo "Log: ${var}"
    sed -i 's/.*_verbose,exec,/\ndnnl_verbose,exec,/' ${var}
    sed -i '/^[[:space:]]*$/d' ${var}
    # get verbose iter
    inf_iters=$(grep 'Iteration: .*, inference time:' ${var} |wc -l)
    verbose_start=$(echo |awk -v inf_iters=${inf_iters} '{print int(inf_iters/2)}')
    verbose_end=$(echo |awk -v inf_iters=${inf_iters} '{print int(inf_iters/2 + 1)}')
    verbose_row_start=$(nl ${var} |grep "Iteration: ${verbose_start}, inference time:" |awk '{print $1}')
    verbose_row_end=$(nl ${var} |grep "Iteration: ${verbose_end}, inference time:" |awk '{print $1}')
    # generate 1 iter verbose log
    output_file="${var}-1iter.log"
    rm -f ${output_file}
    if [ `echo $inf_iters |grep '[1-9]' |wc -l` -ne 0 ];then
        nl ${var} |awk -v verbose_row_start=${verbose_row_start} -v verbose_row_end=${verbose_row_end} '{
            if($1 >= verbose_row_start && $1 <= verbose_row_end){
                print $0;
            }
        }' |sed 's/.*_verbose,exec,/dnnl_verbose,exec,/' |tee -a ${output_file}
    fi
    echo "Output: ${output_file}"
done
