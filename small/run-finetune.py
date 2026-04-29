import os, glob, logging, sys, warnings, time
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
from subprocess import run
import torch
################ INPUT #################
# Check number of cpus/threads availables. If not specified, use all available

try:
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    print('Running using: ',int(os.environ['OMP_NUM_THREADS']), ' threads')
except KeyError:
    torch.set_num_threads(os.cpu_count())
    print('Running using: ',os.cpu_count(), ' threads')



def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()
a = time.time()
train_mace('config.yaml')
b = time.time()

print('Total training time: ', b-a, ' s')
