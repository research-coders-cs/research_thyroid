# research_thyroid

PyTorch based [WSDAN](https://arxiv.org/abs/1901.09891v2) for thyroid image analysis that can be optionally facilitated by Doppler images.

## Credits

Source code in this repository is based on code derived from

- https://github.com/GuYuc/WS-DAN.PyTorch
- https://github.com/wvinzh/WS_DAN_PyTorch
- https://github.com/digitake/research_thyroid
- [WSDAN_Pytorch_Revised_v1_01_a.ipynb](https://colab.research.google.com/drive/1LN4KjBwtq6hUG42LtSLCmIVPasehKeKq)
- [Compare.ipynb](https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk)

## Testing in terminal (`ubuntu-22.04` LTS)

### Prerequisites

```
$ sudo apt install nvidia-cuda-toolkit
$ sudo apt install nvidia-cudnn && sudo /usr/sbin/update-nvidia-cudnn -u
$ pip install pipenv
```

### Setting up the repository

```
$ git clone https://github.com/research-coders-cs/research_thyroid
$ cd research_thyroid
$ pipenv install  # set up the pre-configured python3 virtualenv
```

### Running the tests

```
$ make test  # run 'main.py'

...

@@ torch.__version__: 1.13.1+cu117
@@ demo_thyroid_train(): ^^
@@ device: cpu

...

INFO:root:WSDAN: using densenet as feature extractor, num_classes: 2, num_attentions: 32
INFO:root:WSDAN: All params loaded
INFO:root:Network loaded from ./output/demo_thyroid_train/densenet_250_8_lr-1e5_n4_80.000

...

@@ ======== print_auc(results, enable_plot=0)
@@ y_b: [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
@@ y_m: [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
@@ roc_auc_b: 0.8500000000000001
@@ roc_auc_m: 0.84
@@ demo_thyroid_test(): vv
```
