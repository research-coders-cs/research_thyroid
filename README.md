# Thyroid research

WIP: unify existing codebases

## Testing in terminal (ubuntu-22.04)

### Prerequisites
```
$ sudo apt install nvidia-cuda-toolkit
$ sudo apt install nvidia-cudnn && sudo /usr/sbin/update-nvidia-cudnn -u
$ pip install pipenv
```

### Workflows
```
$ git clone https://github.com/research-coders-cs/research_thyroid
$ cd research_thyroid
$ pipenv install  # set up the pre-configured python3 environment
$ make test       # run 'main.py'

...

@@ torch.__version__: 1.13.1+cu117
@@ matplotlib.__version__: 3.7.0
@@ device: cpu

...

@@ ======== Calling `net = WSDAN(...)`

  0%|          | 0.00/83.3M [00:00<?, ?B/s]
 36%|███▌      | 30.1M/83.3M [00:00<00:00, 316MB/s]
 89%|████████▉ | 74.3M/83.3M [00:00<00:00, 403MB/s]
100%|██████████| 83.3M/83.3M [00:00<00:00, 396MB/s]
INFO:root:WSDAN: using resnet as feature extractor, num_classes: 2, num_attentions: 16
INFO:root:WSDAN: All params loaded

...

@@ ======== print_auc(results, enable_plot=0)
@@ roc_auc_b: 0.42924528301886794
@@ roc_auc_m: 0.4339622641509434
```
