import random

import numpy as np
import torch
import shutil
from torch import nn, ones, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from .callbacks import Callback


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
        self.data_points = []

    def update(self, val, n=1):
        self.data_points.append(val)
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


class BatchCallback(Callback):
    def __init__(self):
        pass

    def on_batch_start(self, *args):
        pass

    def on_batch_end(self, *args):
        self.progress_bar.update()

    def on_epoch_begin(self, description, total_batches):
        # self.progress_bar.reset(total_batches)
        self.progress_bar = tqdm(total=total_batches, unit=' batches')
        self.progress_bar.set_description(description)

    def on_epoch_end(self, *args):
        self.progress_bar.close()


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, train_ds, val_ds, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        # Use dataset to create dataloader
        self.dataloaders = {
            "train": DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True),
            "val": DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
        }
        self.device = device
        self.best_val_loss = np.inf
        self.best_epoch = 0

    def train_one_batch(self, inputs, labels, callback=None):
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
            with torch.no_grad():  # disable grad (we don't need it to cal loss and acc)
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == labels.data)

            return loss.item(), corrects / inputs.shape[0], preds

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

            return loss.item(), corrects / inputs.shape[0], preds

    def train_epoch(self, callback=None):
        # Set model to be in training mode
        self.model.train()
        batch = 1
        loss_meter = AverageMeter('train_loss')
        acc_meter = AverageMeter('train_acc', fmt=':.2f')

        for inputs, labels, extra in self.dataloaders["train"]:
            # move inputs and labels to target device (GPU/CPU/TPU)
            if self.device:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

            callback and callback.on_batch_start()
            loss, acc, preds = self.train_one_batch(inputs, labels)
            callback and callback.on_batch_end(loss, acc, preds)

            with torch.no_grad():
                loss_meter(loss)
                acc_meter(acc)
                batch += 1

        return loss_meter, acc_meter

    def val_epoch(self, callback=None):
        # Set model to be in trianing mode
        self.model.eval()
        batch = 1
        loss_meter = AverageMeter('val_loss')
        acc_meter = AverageMeter('val_acc', fmt=':.2f')

        with torch.no_grad():
            for inputs, labels, extra in self.dataloaders['val']:
                # move inputs and labels to target device (GPU/CPU/TPU)
                if self.device:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                callback and callback.on_batch_start()
                loss, acc, preds = self.val_one_batch(inputs, labels)
                callback and callback.on_batch_end(loss, acc, preds)

                loss_meter(loss)
                acc_meter(acc)
                batch += 1

        return loss_meter, acc_meter

    def train(self, total_epochs, start_epoch=0, callback=None):
        if callback is None:
            callback = BatchCallback()

        for i in range(start_epoch, total_epochs):
            total_train_batches = len(self.dataloaders["train"])
            total_val_batches = len(self.dataloaders["val"])
            callback and callback.on_epoch_begin(f"Epoch {i + 1}/{total_epochs}:",
                                                 total_train_batches + total_val_batches)

            # 1. train one epoch for entire dataset
            loss, acc = self.train_epoch(callback)

            # 2. validate one epoch for entire dataset (no gradient update)
            val_loss, val_acc = self.val_epoch(callback)
            if val_loss.avg < self.best_val_loss:
                self.best_val_loss = val_loss.avg

            log = f"[{loss}, {acc}] : [{val_loss}, {val_acc}]"
            callback and callback.on_epoch_end(i, val_loss.avg)
            print(log)
            print()

    def try_overfit_model(self, inputs, labels, n_epochs=100):
        self.model.train()

        loss_meter = AverageMeter('train_overfit_loss')

        # Test on 100-epoch with the same data to see if network beable to overfit
        for i in range(n_epochs):
            loss = self.train_one_batch(inputs, labels)
            loss_meter(loss)
            if i % 10 == 0:
                print(f"{i}:{loss:.6f}")

        return loss_meter

    def save_model(self, checkpoint_path, val_loss, epoch=1):
        """
        model_state: checkpoint we want to save
        checkpoint_path: path to save checkpoint
        """
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        # save checkpoint data to the path given, checkpoint_path
        torch.save(checkpoint, checkpoint_path)

    def load_model(self, checkpoint_path):
        """
        checkpoint_path: path to save checkpoint
        model: model that we want to load checkpoint parameters into
        optimizer: optimizer we defined in previous training
        """
        # load check point
        checkpoint = torch.load(checkpoint_path)
        # initialize state_dict from checkpoint to model
        self.model.load_state_dict(checkpoint['model_state'])
        # initialize optimizer from checkpoint to optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        # initialize val_loss from checkpoint to val_loss
        self.best_val_loss = checkpoint['val_loss'].item()
        self.best_epoch = checkpoint['epoch'].item()
