import random

import numpy as np
import torch
import shutil
from torch import nn, ones, optim


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

    #torch.use_deterministic_algorithms(True)


## Credit, from pytorch imagenet example
## https://github.com/pytorch/examples/blob/master/imagenet/main.py

class Metric(object):
    pass


class AverageMeter(Metric):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':.4f'):
        super(AverageMeter, self).__init__()
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

    def __call__(self, batch_score, sample_num=1):
        self.update(batch_score, sample_num)
        return self.avg

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def train_one_batch(m, inputs, labels, optimizer_ft):
    pass


def try_overfit_model(m, inputs, labels, device):
    m.train()

    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)

    # Test on 100-epoch with the same data to see if network beable to overfit
    for i in range(100):
        loss = train_one_batch(m, inputs, labels, optimizer)
        if i % 10 == 0:
            print(f"{i}:{loss:.6f}")


class ProgressMeter(Metric):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def __str__(self):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)


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


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, train_dataloader, val_dataloader, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = {
            "train": train_dataloader,
            "val": val_dataloader
        }
        self.device = device

    def train_one_batch(self, inputs, labels):
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):

            # Raw model prediction of size [batch_size, output_classes]
            outputs = self.model(inputs)

            # batch_loss
            loss = self.criterion(outputs, labels)

            # zero the parameter gradients
            # we do this because the optimizer can accumulate loss across batches.
            # so if we don't want to make a big gradient update(it can overshoot), better to reset every batch.
            self.optimizer.zero_grad()

            # Calculate gradient
            loss.backward()

            # Update weight parameter
            self.optimizer.step()

            # prediction as a class number for each outputs
            with torch.no_grad(): # disable grad (we don't need it to cal loss and acc)
              _, preds = torch.max(outputs, 1)
              corrects = torch.sum(preds == labels.data)

            return loss.item(), corrects/inputs.shape[0]

    def val_one_batch(self, inputs, labels):
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Raw model prediction of size [batch_size, output_classes]
            outputs = self.model(inputs)

            # batch_loss
            loss = self.criterion(outputs, labels)

            # prediction as a class number for each outputs
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == labels.data)

            return loss.item(), corrects / inputs.shape[0]

    def train_epoch(self):
        # Set model to be in training mode
        self.model.train()
        batch = 1
        loss_meter = AverageMeter('train_loss')
        acc_meter = AverageMeter('train_acc')

        for inputs, labels, extra in self.dataloaders["train"]:
            # move inputs and labels to target device (GPU/CPU/TPU)
            if self.device:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

            loss, acc = self.train_one_batch(inputs, labels)

            with torch.no_grad():
                loss_meter(loss)
                acc_meter(acc)
                batch += 1

        return loss_meter, acc_meter

    def val_epoch(self):
        # Set model to be in tranning mode
        self.model.eval()
        batch = 1
        loss_meter = AverageMeter('val_loss')
        acc_meter = AverageMeter('val_acc')

        for inputs, labels, extra in self.dataloaders['val']:
            # move inputs and labels to target device (GPU/CPU/TPU)
            if self.device:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

            loss, acc = self.val_one_batch(inputs, labels)

            with torch.no_grad():
                loss_meter(loss)
                acc_meter(acc)
                batch += 1

        return loss_meter, acc_meter


