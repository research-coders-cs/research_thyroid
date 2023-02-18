import random
import shutil

import numpy as np
import torch
from torch import nn, ones
from torchvision import models

from .callbacks import Callback, BatchCallback
from .model_trainer import ModelTrainer
from .resnet_multichannel import Resnet_multichannel, get_arch


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

    # torch.use_deterministic_algorithms(True)


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


def get_densenet161():
    model_ft = models.densenet161(pretrained=True)
    the_layer = replace_prediction_layer(model_ft, 2)

    print("Last Layer", the_layer)

    return model_ft


def get_densenet121():
    model_ft = models.densenet121(pretrained=True)
    the_layer = replace_prediction_layer(model_ft, 2)

    print("Last Layer", the_layer)

    return model_ft


def get_resnet50_4channel():
    resnet50_4_channel = get_arch(50, 4)
    model_ft = resnet50_4_channel(pretrained=True)
    the_layer = replace_prediction_layer(model_ft, 2)

    print("Last Layer", the_layer)

    return model_ft

def get_resnet101_4channel():
    resnet101_4_channel = get_arch(101, 4)
    model_ft = resnet101_4_channel(pretrained=True)
    the_layer = replace_prediction_layer(model_ft, 2)

    print("Last Layer", the_layer)

    return model_ft

def get_resnet152_4channel():
    resnet152_4_channel = get_arch(152, 4)
    model_ft = resnet152_4_channel(pretrained=True)
    the_layer = replace_prediction_layer(model_ft, 2)

    print("Last Layer", the_layer)

    return model_ft
