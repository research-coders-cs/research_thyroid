from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .metric import AverageMeter, TopKAccuracyMetric
from .callback import ModelCheckpoint

import logging

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

def training(device, net, batch_size, train_loader, validate_loader, logs, start_epoch, total_epochs):

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


    

