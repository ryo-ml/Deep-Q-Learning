import logging

def get_logger(name: str = __name__) -> logging.Logger:
    # Handlers and format are configured by Hydra (hydra.job_logging in config.yaml):
    # train.py logs to the console and <output_dir>/train.log, eval.py to the console only.
    return logging.getLogger(name)
