import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskSimilarity(nn.Module):
    def __init__(self, task_dim=128):
        super().__init__()
        self.task_dim = task_dim
        
    def compute_similarity(self, task_a, task_b):
        return F.cosine_similarity(task_a, task_b, dim=-1)
        
    def find_similar_tasks(self, query_task, memory, top_k=5):
        if not memory:
            return []
            
        task_embs = torch.stack([m['task_emb'] for m in memory])
        query_exp = query_task.unsqueeze(0).expand(task_embs.size(0), -1)
        
        sims = self.compute_similarity(query_exp, task_embs)
        top_k = min(top_k, len(memory))
        
        values, indices = torch.topk(sims, top_k)
        return indices.tolist()
