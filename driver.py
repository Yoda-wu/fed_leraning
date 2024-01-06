import subprocess
import sys
import os
import time

from utils.config_parser import load_yml_conf


def process_cmd(yaml_file):
    yaml_conf = load_yml_conf(yaml_file)
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = yaml_conf["job_name"]
    cmd = ""
    log_path = os.path.join(current_path, "log")
    working_dir = os.path.abspath(os.path.join(current_path, yaml_conf["framework"]))
    # 启动server端
    if yaml_conf['framework'] == 'flower':
        cmd = ['python',
               'server.py',
               '--server-address', f'{yaml_conf["server_address"]}',
               '--algorithm', f'{yaml_conf["algorithm"]}',
               '--client-selection', f'{yaml_conf["client_selection"]}',
               '--client-number', f'{yaml_conf["client_number"]}',
               '--gpu', f'{yaml_conf["gpu"]}',
               '--batch-size', f'{yaml_conf["batch_size"]}',
               '--dataset', f'{yaml_conf["dataset"]}',
               '--model', f'{yaml_conf["model"]}',
               '--num-round', f'{yaml_conf["num_round"]}',
               '--frac-clients', f'{yaml_conf["frac-clients"]}',
               '--available-clients',f'{yaml_conf["available_clients"]}'
               ]
    # elif ....


        with open(f"{log_path}/{job_name}_server_logging", "w") as f:
            server_process = subprocess.Popen(cmd, cwd=working_dir, stdout=f, stderr=f, shell=False)
            pid = server_process.pid
            print(f"federate learning  server_process  is running pid is {pid} RUN kill -9 {pid} to stop the job")
        # 启动client端
        print('wait for the server to launch, then will launch clients.....')
        time.sleep(5)
        node_id = 0
        working_dir = os.path.abspath(os.path.join(current_path, f'{yaml_conf["framework"]}/client'))
        for c in range(yaml_conf['client_number']):
            worker_cmd = [
                          "python",
                          "fedavg_client.py",
                          "--node-id",  f"{node_id}",
                          "--server-address", f"{yaml_conf['server_address']}",
                          "--client-number", f"{yaml_conf['client_number']}",
                          "--batch-size", f"{yaml_conf['batch_size']}",
                          "--dataset", f"{yaml_conf['dataset']}",
                          "--model",  f"{yaml_conf['model']}",
                          ]

            with open(f"{log_path}/{job_name}_client_{node_id}_logging", "w") as f:
                client_process = subprocess.Popen(worker_cmd, cwd=working_dir, stdout=f, stderr=f, shell=False)
                pid = client_process.pid
                print(f"federate learning  client_process  is running pid is {pid} RUN kill -9 {pid} to stop the job")
            node_id += 1
        # with open(f"{log_path}/{job_name}_server_logging", "w") as f:
        #     process = subprocess.Popen(f"{cmd}", cwd=working_dir, stdout=f, stderr=f, shell=True)
        #     pid = process.pid
        #     print(f"federate learning process is running pid is {pid} RUN kill -9 {pid} to stop the job")
        # process = subprocess.Popen(f"{cmd}", cwd=working_dir, stderr=subprocess.STDOUT, shell=True)
        # pid = process.pid
        # print(f"federate learning process is running pid is {pid} RUN kill -9 {pid} to stop the job")

        # process.wait()
    elif yaml_conf['framework'] == 'fedml':
        cmd = ["python", "server.py", "--cf", "../../conf/fedml.yaml", "--rank", "0", "--role", "server"]
        working_dir = os.path.abspath(os.path.join(current_path, f'{yaml_conf["framework"]}/server'))
        with open(f"{log_path}/{job_name}_server_logging", "w") as f:
            server_process = subprocess.Popen(cmd, cwd=working_dir, stdout=f, stderr=f, shell=False)
            pid = server_process.pid
            print(f"federate learning  server_process  is running pid is {pid} RUN kill -9 {pid} to stop the job")
        rank = 1
        working_dir = os.path.abspath(os.path.join(current_path, f'{yaml_conf["framework"]}/client'))

        for _ in range(yaml_conf['client_number']):
            worker_cmd = ["python", "client.py", "--cf", "../../conf/fedml.yaml", "--rank", f"{rank}", "--role", "client"]
            with open(f"{log_path}/{job_name}_server_logging", "w") as f:
                client_process = subprocess.Popen(cmd, cwd=working_dir, stdout=f, stderr=f, shell=False)
                pid = server_process.pid
                print(f"federate learning  client_process  is running pid is {pid} RUN kill -9 {pid} to stop the job")
            rank += 1

    print(f"{job_name} is running, please check your logs {log_path}/{job_name}_server/client_logging")
    print("finish start a job. hope everything is good")

def plot_res(job_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_path, "log")
    log_file = f"{log_path}/{job_name}__logging"
    print(log_file)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'start':
            process_cmd(sys.argv[2])
        elif sys.argv[1] == 'plot':
            plot_res(sys.argv[2])
    else:
        print("\033[0;32mUsage:\033[0;0m\n")
        print("start $PATH_TO_CONF_YML      # To run a job")
        print("plot $JOB_NAME               # plot the res of a job")
