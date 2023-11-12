from .peft import load_run_config, do_train
import os
import sleep

def stall():
   sleep(3600)

def do_run():
    config = load_run_config()
    do_train(config)   

def map_env():
    env = os.environ.lookup('env')
    if env == "test":
        return stall
    else
        return do_run

if __name__ == " __main__":
    executor_func = map_env()
    executor_func()
    
