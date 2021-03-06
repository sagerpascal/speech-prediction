import os
import torch.distributed as dist
import datetime


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=3000), rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
