import torch
from torch import nn
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from .metric import AverageMeter, TopKAccuracyMetric
from .augment import batch_augment
from .callback import ModelCheckpoint

import numpy as np
import logging
import os
import time
import gc

from tqdm import tqdm
#from tqdm.notebook import tqdm

##################################
# Activation
##################################

# General loss functions
def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

# @@ not used at the moment
# class SaveFeatures():
#     features=None
#     def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
#     def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
#     def remove(self): self.hook.remove()
#
# def detect_dropler(img_file):
#     img = cv2.imread(img_file)
#     if len(img.shape) < 3:
#         #print('gray_scale')
#         img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
#
#     green_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
#     green_image[:,:] = img[:,:,1]
#     _, threshold = cv2.threshold(green_image, 200, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.imwrite('check.jpg', threshold)
#
#     cntrRect = None
#     maxarea = 0
#     for cnt in contours:
#         hull = cv2.convexHull(cnt,returnPoints = True)
#         #print(cnt)
#         #print(hull)
#         approx = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(cnt, True), True)
#         #cv2.drawContours(img, [approx], 0, (0), 5)
#         if len(approx) == 4:
#             xy = approx.ravel()
#             area = cv2.contourArea(cnt)
#             if maxarea < area:
#                 use = True
#                 for i in range(4):
#                     lx1 = xy[i*2]
#                     ly1 = xy[i*2+1]
#                     if i == 3:
#                         lx2 = xy[0]
#                         ly2 = xy[1]
#                     else:
#                         lx2 = xy[(i+1)*2]
#                         ly2 = xy[(i+1)*2+1]
#                     angle = np.absolute(np.arctan2(ly2-ly1, lx2-lx1)) * 180 / np.pi
#                     if angle > 45 and angle < 135:
#                         angle = angle - 90
#                     if angle >= 135:
#                         angle = angle - 180
#                     if angle > 3:
#                         use = False
#                         break
#
#                 if use:
#                     maxarea = area
#                     cntrRect = approx
#
#     if cntrRect is not None:
#         xy = cntrRect.ravel()
#         x = [xy[0], xy[2], xy[4], xy[6]]
#         y = [xy[1], xy[3], xy[5], xy[7]]
#         min_x = float(min(x))
#         max_x = float(max(x))
#         min_y = float(min(y))
#         max_y = float(max(y))
#         x = [min_x, min_y, max_x, max_y]
#         return x
#     return None


def train(device, logs, data_loader, net, feature_center, optimizer, pbar):  # @@

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()

    example_ct = 0
    for i, (X, y, p) in enumerate(data_loader):
        optimizer.zero_grad()

        # print("i :", i)
        print("X ", X[0].shape)


        # obtain data for training
        X = X.to(device)
        y = y.to(device)

        ##################################
        # Raw Image
        ##################################
        # raw images forward
        y_pred_raw, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = functional.normalize(feature_center[y], dim=-1)
        feature_center[y] += 5e-2 * (feature_matrix.detach() - feature_center_batch)


        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.7, 0.95), padding_ratio=0.1)

        # crop images forward
        y_pred_crop, _, _ = net(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))


        # drop images forward
        y_pred_drop, _, _ = net(drop_images)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                        cross_entropy_loss(y_pred_crop, y) / 3. + \
                        cross_entropy_loss(y_pred_drop, y) / 3. + \
                        center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_crop, y)
            epoch_drop_acc = drop_metric(y_pred_drop, y)

        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}), Crop Acc ({:.2f}), Drop Acc ({:.2f})'.format(
            epoch_loss, epoch_raw_acc[0],
            epoch_crop_acc[0], epoch_drop_acc[0])

        writer.add_scalar("Loss/train", epoch_loss, i)
        writer.add_scalar('Acc(Raw)/train', epoch_raw_acc[0], i)
        writer.add_scalar('Acc(Crop)/train', epoch_crop_acc[0], i)
        writer.add_scalar('Acc(Drop)/train', epoch_drop_acc[0], i)

        example_ct += len(X)
        metrics = {
            #"train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
            "train/example_ct": example_ct,
            "train/loss": epoch_loss,
            "train/raw_acc": epoch_raw_acc[0],
            "train/crop_acc": epoch_crop_acc[0],
            "train/drop_acc": epoch_drop_acc[0],
        }

        #@@ wandb.log(metrics)

        pbar.update()
        pbar.set_postfix_str(batch_info)
        torch.cuda.empty_cache()

    # end of this epoch
    logs['train/{}'.format(loss_container.name)] = epoch_loss
    logs['train/raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train/crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train/drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train/info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(device, logs, data_loader, net, pbar):  # @@

    # metrics initialization
    val_loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
      for i, (X, y, p) in enumerate(data_loader):
          # obtain data
          X = X.to(device)
          y = y.to(device)

          ##################################
          # Raw Image
          ##################################
          y_pred_raw, _, attention_map = net(X)

          ##################################
          # Object Localization and Refinement
          ##################################
          crop_images = batch_augment(X, attention_map, mode='crop', theta=(0.7, 0.95), padding_ratio=0.05)
          y_pred_crop, _, _ = net(crop_images)

          ##################################
          # Final prediction
          ##################################
          y_pred = (y_pred_raw + y_pred_crop) / 2.

          #print("Y_Prediction vs Y_True")
          pred_labels = torch.argmax(y_pred, axis=1)
          pairs = torch.stack((pred_labels, y), dim=1).cpu()
          pairs_with_path = list(zip(pairs, p))
          for (pair, path) in pairs_with_path:
              if (pair[0] - pair[1]) != 0:
                  top_misclassified[path] = top_misclassified.get(path,0) + 1

          # loss
          batch_loss = cross_entropy_loss(y_pred, y)
          epoch_loss = val_loss_container(batch_loss.item())

          # metrics: top-1,5 error
          epoch_acc = raw_metric(y_pred, y)

          probs_raw = softmax(y_pred_raw.cpu().numpy())
          probs_crop = softmax(y_pred_crop.cpu().numpy())

          #@@ table = wandb.Table(columns=["image", "attn", "true_label", "pred_label", "malignancy(%)", "pred_with_crop"])
          for img, attn, true_lab, pred_lab, prob, prob_c in zip(X, attention_map, y, pred_labels, probs_raw, probs_crop):
              a = img[0].cpu().numpy()
              b = attn[0].cpu().numpy()
              p = softmax(prob)
              pp = softmax(prob_c)
              tlab = f'{ "malignant" if true_lab else "benign" }'
              plab = f'{ "malignant" if pred_lab else "benign" }'

              #print(true_lab, tlab, ":", pred_lab, plab)

              #@@ table.add_data(wandb.Image(a*255),
              #                 wandb.Image(b*255),
              #                 tlab,
              #                 plab,
              #                 f"{p[1]*100:.2f}%",
              #                 f"{pp[1]*100:.2f}%")

          #@@ wandb.log({"predictions":table}, commit=False)
          torch.cuda.empty_cache()

    epoch = logs['epoch'] if 'epoch' in logs else 0
    loss = {
        'train': logs['train/loss'],
        'val': epoch_loss,
    }
    raw_acc = {
        'train': logs['train/raw_topk_accuracy'],
        'val': logs['val/topk_accuracy'],
    }
    crop_drop_acc = {
        'crop': logs['train/crop_topk_accuracy'][0],
        'drop': logs['train/drop_topk_accuracy'][0]
    }

    writer.add_scalars("Loss", loss, epoch)
    writer.add_scalars('Acc', raw_acc, epoch)
    writer.add_scalars('Acc/Crop-Drop', crop_drop_acc, epoch)

    writer.flush()

    # end of validation
    logs['val/{}'.format(loss_container.name)] = epoch_loss
    logs['val/{}'.format(raw_metric.name)] = epoch_acc
    # print("Update val_{}=".format(raw_metric.name), epoch_acc)
    end_time = time.time()

    metrics = {
        "val/epoch_loss": epoch_loss,
        "val/epoch_acc": epoch_acc,
        "val/top_misclassified": top_misclassified
    }
    #@@ wandb.log(metrics)

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f})'.format(epoch_loss, epoch_acc[0])
    pbar.set_postfix_str('{}, {}'.format(logs['train/info'], batch_info))

    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


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
             optimizer, scheduler, run_name, logs, start_epoch, total_epochs,
             savepath='.'):

    callback_monitor = 'val/{}'.format(raw_metric.name)
    callback = ModelCheckpoint(
        savepath=os.path.join(savepath, run_name),
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

        train(device, logs, train_loader, net, feature_center, optimizer, pbar)

        validate(device, logs, validate_loader, net, pbar)

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
    return callback.get_savepath_with_best_score()  # @@
