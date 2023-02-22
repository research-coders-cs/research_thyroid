
import torch


import digitake






"""## 2.2 Global config for training environment and reproducibility"""

USE_GPU = False#True
digitake.model.set_reproducible(2565)

if USE_GPU:
    # GPU settings
    assert torch.cuda.is_available(), "Don't forget to turn on gpu runtime!"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print("@@ device:", device)



if __name__ == '__main__':
    pass