import random

import numpy as np
import torch
import shutil
from torch import nn, ones


def check_model_last_layer(m):
    layers = list(m.children())
    total_layer = len(layers)
    last_layer = layers[-1]
    is_last_layer_linear = type(last_layer) is nn.Linear
    print(f"{total_layer} - {last_layer} is Linear? : {is_last_layer_linear}")
    return is_last_layer_linear


def get_last_linear_layer(m):
    layers = list(m.children())
    last_layer = layers[-1]
    if type(last_layer) is nn.Linear:
        return last_layer
    else:
        return get_last_linear_layer(last_layer)


def replace_prediction_layer(model, n):
    """
    Inplace replacement for the last layer out_features
    :param model: to be replace
    :param n: num_features
    :return: Last lasyer or raise Exception if replacement is failed
    """
    ldn = get_last_linear_layer(model)
    if ldn is not None:
        ldn.out_features = n
        # We have to reset the Weight matrix and bias as well, or it will not change
        ldn.weight = nn.Parameter(ones(ldn.out_features, ldn.in_features))
        ldn.bias = nn.Parameter(ones(ldn.out_features))
        # Re-Randomize bias and weight
        ldn.reset_parameters()
        return ldn
    else:
        raise Exception("Last prediction layer not found")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def set_reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms()


## Credit, from pytorch imagenet example
## https://github.com/pytorch/examples/blob/master/imagenet/main.py

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
