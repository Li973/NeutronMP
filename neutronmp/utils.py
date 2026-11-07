import math
import os

import numpy as np
import torch
import torch.nn.functional as F

EPSILON = 1.0 - math.log(2.0)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def modified_cross_entropy(logits, labels):
    if labels.dim() == 2:
        labels = labels.squeeze(1)
    ce = F.cross_entropy(logits, labels, reduction="none")
    ce = torch.clamp(ce, min=1e-9)
    return (torch.log(EPSILON + ce) - math.log(EPSILON)).mean()
