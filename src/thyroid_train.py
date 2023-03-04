from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .metric import AverageMeter, TopKAccuracyMetric
from .callback import ModelCheckpoint

import logging

from tqdm import tqdm
#from tqdm.notebook import tqdm

##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)

##################################
# Loss
##################################

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

##################################
# ModelCheckpoint
##################################

# loss and metric
loss_container = AverageMeter(name='loss')
val_loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric()
crop_metric = TopKAccuracyMetric()
drop_metric = TopKAccuracyMetric()


top_misclassified = {}
writer = SummaryWriter()

def training(device, net, feature_center, batch_size, train_loader, validate_loader,
             logs, start_epoch, total_epochs, optimizer):

    callback_monitor = 'val/{}'.format(raw_metric.name)
    callback = ModelCheckpoint(
        # savepath=os.path.join(f'./{name}'),  # @@
        savepath=f'./xxxx',  # @@ !!!!!!!!!!
        monitor=callback_monitor,
        mode='max')

    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    #

    for epoch in range(start_epoch, start_epoch + total_epochs):
        print(('#' * 10), 'epoch ', str(epoch + 1), ('#' * 10))

        callback.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, total_epochs))

        train(
            logs=logs,
            data_loader=train_loader,
            net=net,
            feature_center=feature_center,
            optimizer=optimizer,
            pbar=pbar
        )

        validate(
            logs=logs,
            data_loader=validate_loader,
            net=net,
            pbar=pbar
        )

        # Checkpoints
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val/loss'])
        else:
            scheduler.step()

        callback.on_epoch_end(logs, net, feature_center=feature_center)

        #@@wandb.log(logs)
        pbar.close()
        writer.flush()

        gc.collect()
        torch.cuda.empty_cache()

    #@@wandb.finish()
