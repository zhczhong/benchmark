@NonCPS
def jsonParse(def json) {
    new groovy.json.JsonSlurperClassic().parseText(json)
}

SUB_NODE_LABEL = ''
if ('SUB_NODE_LABEL' in params) {
    echo "SUB_NODE_LABEL in params"
    if (params.SUB_NODE_LABEL != '') {
        SUB_NODE_LABEL = params.SUB_NODE_LABEL
    }
}
echo "SUB_NODE_LABEL: $SUB_NODE_LABEL"

CONDA_PATH = ''
if ('CONDA_PATH' in params) {
    echo "CONDA_PATH in params"
    if (params.CONDA_PATH != '') {
        CONDA_PATH = params.CONDA_PATH
    }
}
echo "CONDA_PATH: $CONDA_PATH"

VIRTUAL_ENV = 'oob'
if ('VIRTUAL_ENV' in params) {
    echo "VIRTUAL_ENV in params"
    if (params.VIRTUAL_ENV != '') {
        VIRTUAL_ENV = params.VIRTUAL_ENV
    }
}
echo "VIRTUAL_ENV: $VIRTUAL_ENV"

PYTORCH_PRETRAIN_DIR1 = ''
if ('PYTORCH_PRETRAIN_DIR1' in params) {
    echo "PYTORCH_PRETRAIN_DIR1 in params"
    if (params.PYTORCH_PRETRAIN_DIR1 != '') {
        PYTORCH_PRETRAIN_DIR1 = params.PYTORCH_PRETRAIN_DIR1
    }
}
echo "PYTORCH_PRETRAIN_DIR1: $PYTORCH_PRETRAIN_DIR1"

PRECISION = 'float32'
if ('PRECISION' in params) {
    echo "PRECISION in params"
    if (params.PRECISION != '') {
        PRECISION = params.PRECISION
    }
}
echo "PRECISION: $PRECISION"

RUN_PERF = '1'
if ('RUN_PERF' in params) {
    echo "RUN_PERF in params"
    if (params.RUN_PERF != '') {
        RUN_PERF = params.RUN_PERF
    }
}
echo "RUN_PERF: $RUN_PERF"


COLLECT_DNNL_VERBOSE = '0'
if ('COLLECT_DNNL_VERBOSE' in params) {
    echo "COLLECT_DNNL_VERBOSE in params"
    if (params.COLLECT_DNNL_VERBOSE != '') {
        COLLECT_DNNL_VERBOSE = params.COLLECT_DNNL_VERBOSE
    }
}
echo "COLLECT_DNNL_VERBOSE: $COLLECT_DNNL_VERBOSE"


MODEL_NAME = ''
if ('MODEL_NAME' in params) {
    echo "MODEL_NAME in params"
    if (params.MODEL_NAME != '') {
        MODEL_NAME = params.MODEL_NAME
    }
}
echo "MODEL_NAME: $MODEL_NAME"



def cleanup(){
    try {
        sh '''#!/bin/bash 
        set -x
        cd $WORKSPACE
        rm -rf *
        lscpu
        '''
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

node(SUB_NODE_LABEL){

    
    dir("oob_perf") {
        def folder = new File('$WORKSPACE')
        if (!folder.exists()){
            checkout scm
        }
        else{
            println("the file is exists! don't download agin")
            sh 'git show'
        }
    }

    try{
        
        def inputJson = jsonParse(readFile("$WORKSPACE/oob_perf/PT/model.json"))
        model_path = inputJson[MODEL_NAME]["model_path"]
        
        stage("RUN MODEL"){
            
            withEnv(["model_name=${MODEL_NAME}", "model_path=${model_path}","CONDA_PATH=${CONDA_PATH}","conda_env=${VIRTUAL_ENV}", \
            "run_perf=${RUN_PERF}","collect_dnnl_verbose=${COLLECT_DNNL_VERBOSE}","precision=${PRECISION}","pytorch_pretrain_dir=${PYTORCH_PRETRAIN_DIR1}"])
            {
                sh '''
                    #!/bin/bash
                    set +e
                    . ${CONDA_PATH}
                    conda activate ${conda_env}
                    CUR_PATH=`pwd`
                    cd $CUR_PATH/oob_perf
                    if [ "`ls -A $CUR_PATH/oob_perf/PT/gpt-2`"="" ];then
                        git submodule update --init PT
                    fi
                    pip install -r  $CUR_PATH/oob_perf/PT/requirements.txt
                    which python
                    
                    echo $CUR_PATH
                    workspace="$CUR_PATH/OOB_PT_Logs/"
                    oob_home_path=$CUR_PATH
                    if [ -d ${workspace} ];then
                        echo "Will runing ${model_name}"
                    else
                        mkdir -p ${workspace}
                    fi
                    echo $conda_env
                    pwd

                    ls
                    cd PT/${model_path}
                    bash ./auto_benchmark.sh --workspace=${workspace} --run_perf=${run_perf} \
                        --collect_dnnl_verbose=${collect_dnnl_verbose} --precision=${precision} --pytorch_pretrain_dir=${pytorch_pretrain_dir}
                '''
             }
        }

    }catch(e){
        currentBuild.result = "FAILURE"
        throw e
    }finally{
        // save log files
        stage("Archive Artifacts") {
            archiveArtifacts artifacts: "**/OOB_PT_Logs/**", excludes: null
            fingerprint: true
        }
    }
}