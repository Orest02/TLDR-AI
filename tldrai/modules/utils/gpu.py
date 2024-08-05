import logging

import torch


def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3  # Convert to GB
        total_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )  # Convert to GB
        free_memory = total_memory - reserved

        logging.info(f"Allocated memory: {allocated:.2f} GB")
        logging.info(f"Reserved memory: {reserved:.2f} GB")
        logging.info(f"Total memory: {total_memory:.2f} GB")
        logging.info(f"Free memory: {free_memory:.2f} GB")

        return free_memory
    else:
        logging.info("CUDA is not available.")
        return 0
