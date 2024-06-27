
#---- ^^
import os
from os.path import exists
import torch
# import torch.nn as nn
# from torch.nn.functional import log_softmax, pad
# import math
# import copy
# import time
# from torch.optim.lr_scheduler import LambdaLR
# import pandas as pd
# import altair as alt
# from torchtext.data.functional import to_map_style_dataset
# from torch.utils.data import DataLoader

# import GPUtil

# from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

# Set to False to skip notebook execution (e.g. for debugging)
import warnings
warnings.filterwarnings("ignore")
#---- $$

print("@@ torch.__version__:", torch.__version__)  # torch 1.11 (enforced by `torchdata==0.3.0`)

