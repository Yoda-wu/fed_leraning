import subprocess
import sys
import os
from utils.config_parser import load_yml_conf


def process_cmd(yaml_file):
    yaml_conf = load_yml_conf(yaml_file)
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = yaml_conf["job_name"]
    cmd = ""
    if yaml_conf['framework'] == 'flower':
        cmd = (f'python {current_path}/{yaml_conf["framework"]}/main.py --algorithm {yaml_conf["algorithm"]} '
               f'--client-selection {yaml_conf["client_selection"]} '
               f'--client-number {yaml_conf["client_number"]}'
               f'--device {yaml_conf["device"]}'
               f'--batch-size {yaml_conf["batch_size"]}'
               f'--dataset {yaml_conf["dataset"]}'
               f'--model {yaml_conf["model"]}'
               f'--num-round {yaml_conf["num_round"]}'
               f'--frac-clients {yaml_conf["frac-clients"]}'
               f'--available-clients {yaml_conf["available_clients"]}'
               )
    # elif ....

    log_path = os.path.join(current_path, "log")

    with open(f"{log_path}/{job_name}_logging", "a+") as f:
        process = subprocess.Popen(f"{cmd}", stdout=f, stderr=f)
        pid = process.pid
        print(f"federate learning process is running pid is {pid} RUN kill -9 {pid} to stop the job")

    print(f"{job_name} is running, please check your logs {log_path}/{job_name}__logging")
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
