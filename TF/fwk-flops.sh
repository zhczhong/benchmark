#!/bin/bash
# caculate framwork level flops

if [ "${log_dir}" == "" ] || [ ! -d ${log_dir} ];then
    log_dir="${PWD}/logs"
    mkdir -p ${log_dir}
fi
summary_log="${log_dir}/fwk-efficiency.log"

function main {
    input_log="$1"
    # shape log
    shape_file="/tmp/shape-$(date +%s).log"
    onednn_time=$(grep -v 'Total:' ${input_log} |awk 'BEGIN{sum=0}{sum+=$2}END{print sum}')
    # exclude for dw/group conv and matmul gemmv
    cat ${input_log} |awk '{
        split($1, primitive, ",");
        prim = primitive[1];
        if(prim == "convolution") {
            shape = primitive[4];
            if(!(shape ~/g[0-9]*mb/)) {
                print $0;
            }
        }
    }' > ${shape_file}
    cat ${input_log} |awk '{
        split($1, primitive, ",");
        prim = primitive[1];
        if(prim == "matmul") {
            shape = primitive[4];
            split(shape, column, ":");
            split(column[1], column1, "x");
            split(column[2], column2, "x");
            split(column[3], column3, "x");
            m = column1[length(column1) - 1];
            k = column1[length(column1)];
            n = column2[length(column2)];
            if(m!=1 && k!=1 && n!=1) {
                print $0;
            }
        }
    }' >> ${shape_file}
    cat ${input_log} |awk '{
        split($1, primitive, ",");
        prim = primitive[1];
        if(prim == "inner_product") {
            shape = primitive[4];
            split(shape, column, "mb");
            split(column[2], column1, "ic");
            split(shape, column, "ic");
            split(column[2], column2, "oc");
            split(shape, column, "oc");
            split(column[2], column3, "[^0-9]");
            m = column1[1];
            k = column2[1];
            n = column3[1];
            if(m!=1 && k!=1 && n!=1) {
                print $0;
            }
        }
    }' >> ${shape_file}
    comp_time=$(awk 'BEGIN{sum=0}{sum+=$2}END{print sum}' ${shape_file})
    conv_time=$(awk 'BEGIN{sum=0}{if($1 ~/convolution/){sum+=$2}}END{print sum}' ${shape_file})
    matmul_time=$(awk 'BEGIN{sum=0}{if($1 ~/matmul/){sum+=$2}}END{print sum}' ${shape_file})
    ip_time=$(awk 'BEGIN{sum=0}{if($1 ~/inner_product/){sum+=$2}}END{print sum}' ${shape_file})
    # efficiency
    get_efficiency
}

function get_efficiency {
    # for computation
    awk -M -v PREC=1024 -v model_name=${model_name} -v end2end_time=${end2end_time} -v onednn_time=${onednn_time} -v comp_time=${comp_time} \
         -v conv_time=${conv_time} -v matmul_time=${matmul_time} -v ip_time=${ip_time} 'BEGIN{
        printf("Model,Category,Primitive,Impl,DataType,Shape,Duration,Call,GFLOPs,GFLOPS,SPRGFLOPS,Efficiency\n");
        comp_min_shape = "";
        comp_min_eff = 1;
        comp_num = 0;
        comp_sum = 0;
        comp_mul = 1;
        comp_wei_eff = 0;
        comp_avg_eff = 0;
        comp_geo_eff = 0;
        conv_min_shape = "";
        conv_min_eff = 1;
        conv_num = 0;
        conv_sum = 0;
        conv_mul = 1;
        conv_wei_eff = 0;
        conv_avg_eff = 0;
        conv_geo_eff = 0;
        matmul_min_shape = "";
        matmul_min_eff = 1;
        matmul_num = 0;
        matmul_sum = 0;
        matmul_mul = 1;
        matmul_wei_eff = 0;
        matmul_avg_eff = 0;
        matmul_geo_eff = 0;
        ip_min_shape = "";
        ip_min_eff = 1;
        ip_num = 0;
        ip_sum = 0;
        ip_mul = 1;
        ip_wei_eff = 0;
        ip_avg_eff = 0;
        ip_geo_eff = 0;
    }{
        split($1, primitive, ",");
        prim = primitive[1];
        impl = primitive[2];
        data_type = primitive[3];
        shape = primitive[4];
        // gflops
        if(prim == "convolution") {
            split(shape, column, "^g");
            split(column[2], column1, "[^0-9]");
            g = column1[1];
            split(shape, column, "mb");
            split(column[2], column1, "[^0-9]");
            mb = column1[1];
            split(shape, column, "ic");
            split(column[2], column1, "[^0-9]");
            ic = column1[1];
            split(shape, column, "oc");
            split(column[2], column1, "[^0-9]");
            oc = column1[1];
            split(shape, column, "kw");
            split(column[2], column1, "[^0-9]");
            kw = column1[1];
            split(shape, column, "oh");
            split(column[2], column1, "[^0-9]");
            oh = column1[1];
            split(shape, column, "ow");
            split(column[2], column1, "[^0-9]");
            ow = column1[1];
            split(shape, column, "ih");
            split(column[2], column1, "[^0-9]");
            ih = column1[1];
            split(shape, column, "iw");
            split(column[2], column1, "[^0-9]");
            iw = column1[1];
            if(g ~/[1-9]/) {
                GFLOPs = 2 * mb * ic * oc * oh * ow / 1000 / 1000 / 1000;
            }else {
                GFLOPs = 2 * mb * ic * oc * kw * kw * oh * ow / 1000 / 1000 / 1000;
            }
        }
        if(prim == "matmul") {
            split(shape, column, ":");
            split(column[1], column1, "x");
            split(column[2], column2, "x");
            split(column[3], column3, "x");
            m = column1[length(column1) - 1];
            k = column1[length(column1)];
            n = column2[length(column2)];
            if(length(column1) > 2) {
                bs = column1[length(column1) - 2];
                GFLOPs = 2 * m * k * n * bs / 1000 / 1000 / 1000;
            }else {
                GFLOPs = 2 * m * k * n / 1000 / 1000 / 1000;
            }
        }
        if(prim == "inner_product") {
            split(shape, column, "mb");
            split(column[2], column1, "ic");
            split(shape, column, "ic");
            split(column[2], column2, "oc");
            split(shape, column, "oc");
            split(column[2], column3, "[^0-9]");
            mb = column1[1];
            ic = column2[1];
            oc = column3[1];
            GFLOPs = 2 * mb * ic * oc / 1000 / 1000 / 1000;
        }
        // spr gflops
        spr_gflops = 4 * 2.0 * 2 * 2 / 32;
        if(data_type == "int8") {
            spr_gflops = spr_gflops * 4;
        }else if(data_type == "bf16") {
            spr_gflops = spr_gflops * 2;
        }
        if(impl == "amx") {
            spr_gflops = spr_gflops * 512 * 8;
        }else if(impl == "vnni") {
            spr_gflops = spr_gflops * 512 * 4;
        }else if(impl == "avx512") {
            spr_gflops = spr_gflops / 2.0 * 2.4 * 512;
        }else {
            spr_gflops = 0;
        }
        // efficiency
        duration = $2;
        call = $3;
        GFLOPS = GFLOPs / (duration / call / 1000);
        efficiency = GFLOPS / spr_gflops;
        printf("%s,details,%s,%s,%s,%s,%s,%s,%.8f,%.8f,%.2f,%.8f\n",
                model_name, prim, impl, data_type, shape, duration, call, GFLOPs, GFLOPS, spr_gflops, efficiency);
        if(efficiency < conv_min_eff) {
            conv_min_eff = efficiency;
            conv_min_shape = shape;
        }
        if(efficiency < matmul_min_eff) {
            matmul_min_eff = efficiency;
            matmul_min_shape = shape;
        }
        if(efficiency < ip_min_eff) {
            ip_min_eff = efficiency;
            ip_min_shape = shape;
        }
        if(efficiency < comp_min_eff) {
            comp_min_eff = efficiency;
            comp_min_shape = shape;
        }
        if(efficiency ~/[1-9]/) {
            if(prim == "convolution") {
                conv_num ++;
                conv_sum += efficiency;
                conv_mul *= efficiency;
                conv_wei_eff += efficiency * duration / conv_time;
            }
            if(prim == "matmul") {
                matmul_num ++;
                matmul_sum += efficiency;
                matmul_mul *= efficiency;
                matmul_wei_eff += efficiency * duration / matmul_time;
            }
            if(prim == "inner_product") {
                ip_num ++;
                ip_sum += efficiency;
                ip_mul *= efficiency;
                ip_wei_eff += efficiency * duration / ip_time;
            }
            comp_num ++;
            comp_sum += efficiency;
            comp_mul *= efficiency;
            comp_wei_eff += efficiency * duration / comp_time;
        }
    }END{
        printf("Model,Category,End2End,OneDNN,Primitive,PrimTime,MinShape,Minimal,Average,Geomean,Weigthed\n");
        if(conv_num > 0) {
            conv_avg_eff = conv_sum / conv_num;
            conv_geo_eff = conv_mul ^ (1 / conv_num);
            printf("%s,summary,%.3f,%.3f,convolution,%.3f,%s,%.8f,%.8f,%.8f,%.8f\n",
                    model_name, end2end_time, onednn_time, conv_time, conv_min_shape, conv_min_eff, conv_avg_eff, conv_geo_eff, conv_wei_eff);
        }
        if(matmul_num > 0) {
            matmul_avg_eff = matmul_sum / matmul_num;
            matmul_geo_eff = matmul_mul ^ (1 / matmul_num);
            printf("%s,summary,%.3f,%.3f,matmul,%.3f,%s,%.8f,%.8f,%.8f,%.8f\n",
                    model_name, end2end_time, onednn_time, matmul_time, matmul_min_shape, matmul_min_eff, matmul_avg_eff, matmul_geo_eff, matmul_wei_eff);
        }
        if(ip_num > 0) {
            ip_avg_eff = ip_sum / ip_num;
            ip_geo_eff = ip_mul ^ (1 / ip_num);
            printf("%s,summary,%.3f,%.3f,inner_product,%.3f,%s,%.8f,%.8f,%.8f,%.8f\n",
                    model_name, end2end_time, onednn_time, ip_time, ip_min_shape, ip_min_eff, ip_avg_eff, ip_geo_eff, ip_wei_eff);
        }
        if(comp_num > 0) {
            comp_avg_eff = comp_sum / comp_num;
            comp_geo_eff = comp_mul ^ (1 / comp_num);
            printf("%s,summary,%.3f,%.3f,computation,%.3f,%s,%.8f,%.8f,%.8f,%.8f\n",
                    model_name, end2end_time, onednn_time, comp_time, comp_min_shape, comp_min_eff, comp_avg_eff, comp_geo_eff, comp_wei_eff);
        }
    }' ${shape_file} |tee ${summary_log}
}

main $@
