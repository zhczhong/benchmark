import groovy.json.*
NODE_LABEL = 'master'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

SUB_NODE_LABEL = 'ptoob'
if ('SUB_NODE_LABEL' in params) {
    echo "SUB_NODE_LABEL in params"
    if (params.SUB_NODE_LABEL != '') {
        SUB_NODE_LABEL = params.SUB_NODE_LABEL
    }
}
echo "SUB_NODE_LABEL: $SUB_NODE_LABEL"

// first support single sub_node, should support multi sub_node through SUB_NODE_LABEL
SUB_NODE_HOSTNAME = ''
if ('SUB_NODE_HOSTNAME' in params) {
    echo "SUB_NODE_HOSTNAME in params"
    if (params.SUB_NODE_HOSTNAME != '') {
        SUB_NODE_HOSTNAME = params.SUB_NODE_HOSTNAME
    }
}
echo "SUB_NODE_HOSTNAME: $SUB_NODE_HOSTNAME"

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



ALL_MODELS = ''
if ('ALL_MODELS' in params) {
    echo "ALL_MODELS in params"
    if (params.ALL_MODELS != '') {
        ALL_MODELS = params.ALL_MODELS
    }
}
echo "ALL_MODELS: $ALL_MODELS"

// param to count completed models
run_model_counts = 0
run_model_completed_counts = 0

def Run_Models_Jobs()
{

    def jobs = [:]

    
    def MODEL_LIST = ALL_MODELS.split(",")
    run_model_counts = MODEL_LIST.size()
    println("run_model_counts = " + run_model_counts)
    MODEL_LIST.each { case_name ->
        if (case_name != ""){
            List model_params = [
                    string(name: "SUB_NODE_LABEL", value: SUB_NODE_LABEL),
                    string(name: "CONDA_PATH", value: CONDA_PATH),
                    string(name: "VIRTUAL_ENV", value: VIRTUAL_ENV),
                    string(name: "PYTORCH_PRETRAIN_DIR1", value: PYTORCH_PRETRAIN_DIR1),
                    string(name: "PRECISION", value: PRECISION),
                    string(name: "RUN_PERF", value: RUN_PERF),
                    string(name: "MODEL_NAME", value: case_name),
                    string(name: "COLLECT_DNNL_VERBOSE", value: COLLECT_DNNL_VERBOSE)
            ]

            jobs["${case_name}"] = {
                //println("---------${FRAMEWORK}_${case_name}_precision----------")
                sub_jenkins_job = "test_meng_PT_OOB_sub"
                downstreamJob = build job: sub_jenkins_job, propagate: false, parameters: model_params

                catchError {
                    copyArtifacts(
                            projectName: sub_jenkins_job,
                            selector: specific("${downstreamJob.getNumber()}"),
                            filter: 'OOB_PT_Logs/**',
                            fingerprintArtifacts: true,
                            target: "${case_name}",
                            optional: true)

                    // Archive in Jenkins
                    archiveArtifacts artifacts: "${case_name}/**", allowEmptyArchive: true
                    sh """#!/bin/bash
                        if [ -r ${case_name}/OOB_PT_Logs/summary.log ]; then
                            cat ${case_name}/OOB_PT_Logs/summary.log >> ${WORKSPACE}/summary.log
                        else
                            echo "${case_name},failed" >> ${WORKSPACE}/summary.log

                        fi
                    """
                }

                def downstreamJobStatus = downstreamJob.result
                run_model_completed_counts += 1

                if (downstreamJobStatus != 'SUCCESS') {
                    currentBuild.result = "FAILURE"
                }
            }
        }
    }

    return jobs
}

def Status_Check_Job(){
    println("status check job")
    status_check_round = 0
    SUB_NODE_HOSTNAME = SUB_NODE_HOSTNAME.split(',')
    while(run_model_completed_counts < run_model_counts){
        println("run_model_completed_counts = " + run_model_completed_counts)
        println("run_model_counts = " + run_model_counts)
        SUB_NODE_HOSTNAME.each { case_name ->
            withEnv(["sub_host_name=$case_name"]){
                sh'''#!/bin/bash
                    status=`timeout 3 ssh pyuan@${sub_host_name} echo 1`
                    if [ $? != 0 ]; then 
                        echo "reboot"
                    fi
                '''
            }
            status_check_round += 1
            println("status_check_round：" + status_check_round)
            sleep(10)
        }
    }  
}

node(NODE_LABEL){
    try {
        deleteDir()
        dir("oob_perf") {
            checkout scm
        }

        stage("RUN MODELS") {
            sh'''
            echo `pwd`
            '''
            def job_list = [:]
            def model_jobs = Run_Models_Jobs()

            if (model_jobs.size() > 0){
                job_list['Status Check'] = {
                    Status_Check_Job()
                }
                job_list = job_list + model_jobs
            }

            parallel job_list
        }
        
    }catch(Exception e) {
        currentBuild.result = "FAILURE"
        error(e.toString())
    } finally {
        dir(WORKSPACE){
            sh'''#!/bin/bash
                if [ -f summary.log ]; then
                    cp summary.log results.csv
                fi
            '''
        }
        archiveArtifacts '*.log, *.csv'
    }
}