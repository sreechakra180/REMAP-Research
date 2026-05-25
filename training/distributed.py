import torch
import torch.distributed as dist
import os

def setup_distributed(config):
    """Sets up PyTorch distributed data parallel."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=config.get('backend', 'nccl'), rank=rank, world_size=world_size)
        return True
    return False

class GradientAccumulator:
    """Handles gradient accumulation across steps."""
    def __init__(self, accumulation_steps):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_step(self):
        self.current_step += 1
        if self.current_step % self.accumulation_steps == 0:
            self.current_step = 0
            return True
        return False
