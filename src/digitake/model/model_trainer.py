import numpy as np
import torch
from torch.utils.data import DataLoader

from .meter import AverageMeter
from .callbacks import BatchCallback


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, train_ds, val_ds, device=None, shuffle_valset=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        # Use dataset to create dataloader
        self.dataloaders = {
            "train": DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True),
            "val": DataLoader(val_ds, batch_size=16, shuffle=shuffle_valset, num_workers=2, pin_memory=True)
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

            acc = corrects / inputs.shape[0]
            return loss.item(), acc.item(), preds, labels, "train"

    def val_one_batch(self, inputs, labels):

        # We don't need gradient on validation
        with torch.set_grad_enabled(False):
            # Raw model prediction of size [batch_size, output_classes]
            outputs = self.model(inputs)

            # batch_loss
            loss = self.criterion(outputs, labels)

            # prediction as a class number for each outputs
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == labels.data)
            acc = corrects / inputs.shape[0]
            return loss.item(), acc.item(), preds, labels, "val"

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
            loss, acc, preds, labels, phase = self.train_one_batch(inputs, labels)
            callback and callback.on_batch_end(loss, acc, preds, labels, phase)

            with torch.no_grad():
                loss_meter(loss)
                acc_meter(acc)
                batch += 1

        return loss_meter, acc_meter

    def val_epoch(self, callback=None, val_loader=None):
        # Custom vs embedded val set
        val_loader = val_loader or self.dataloaders['val']

        # Set model to be in eval mode
        self.model.eval()
        batch = 1
        loss_meter = AverageMeter('val_loss')
        acc_meter = AverageMeter('val_acc', fmt=':.2f')

        with torch.no_grad():
            for inputs, labels, extra in val_loader:
                # move inputs and labels to target device (GPU/CPU/TPU)
                if self.device:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                callback and callback.on_batch_start()
                loss, acc, preds, labels, phase = self.val_one_batch(inputs, labels)
                callback and callback.on_batch_end(loss, acc, preds, labels, phase)

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

            #log = f"[{loss}, {acc}] : [{val_loss}, {val_acc}]"
            callback and callback.on_epoch_end(i, loss, acc, val_loss, val_acc)
            #print(log)
            #print()

    def eval(self, ext_val_ds, batch_size=16, shuffle=True):
        ext_val = DataLoader(ext_val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
        val_loss, val_acc = self.val_epoch(val_loader=ext_val)

        return (val_loss, val_acc)

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
        self.best_val_loss = checkpoint['val_loss']
        self.best_epoch = checkpoint['epoch']
