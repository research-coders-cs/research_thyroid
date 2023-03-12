import torch
from torch import nn
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from .metric import AverageMeter, TopKAccuracyMetric
from .augment import batch_augment, img_gpu_to_cpu
from .checkpoint import ModelCheckpoint
from .doppler import detect_doppler, get_iou#, plot_comp, get_sample_paths
from .utils import show_data_loader

import cv2
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


class SaveFeatures():  # @@ not used at the moment
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def train(device, logs, train_loader, doppler_train_loader, net, feature_center, optimizer, pbar):

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()

    # @@ !!!!
    to_doppler = {
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0011_2_p0022.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0011_1_p0022.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0076_2_p0152.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0076_1_p0152.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0022_2_p0044.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0022_1_p0044.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule3_0001-0030_c0024_1_p0071.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule3_0001-0030_c0024_2_p0071.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_siriraj_0001-0160_c0128_1_p0088.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_siriraj_0001-0160_c0128_2_p0089.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0008_2_p0016.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0008_1_p0016.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_siriraj_0001-0124_c0110_3_p0257.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_nodule3_0001-0030_c0004_3_p0011.png',
    }

    example_ct = 0
    for idx, (X, y, p) in enumerate(train_loader):
        optimizer.zero_grad()

        print(f"(batch idx={idx}) X[0].shape:", X[0].shape)
        if 0:  # @@ !!!!
            for img_idx, train_img_path in enumerate(p['path']):
                doppler_img_path = to_doppler[train_img_path]
                print(f'@@ {train_img_path} ->\n  {doppler_img_path}')

                train_img = img_gpu_to_cpu(X[img_idx])
                train_img = np.array(train_img).astype(np.uint8).copy()
                #cv2.imwrite(f'train_img_{img_idx}.jpg', train_img)

                img = cv2.imread(doppler_img_path)
                width = int(img.shape[1])
                height = int(img.shape[0])
                print('@@ (doppler) width, height:', width, height)

                temp = detect_doppler(doppler_img_path)
                #                             vvvvvvvvvvvvvvvvvvvvv
                x1_doppler_calc = int(temp[0] * 250. / img.shape[1])
                y1_doppler_calc = int(temp[1] * 250. / img.shape[0])
                x2_doppler_calc = int(temp[2] * 250. / img.shape[1])
                y2_doppler_calc = int(temp[3] * 250. / img.shape[0])
                bbox_doppler = np.array([x1_doppler_calc, y1_doppler_calc, x2_doppler_calc, y2_doppler_calc], dtype=np.float32)
                #@@border_img_doppler = cv2.rectangle(src_doppler, (x1_doppler_calc, y1_doppler_calc), (x2_doppler_calc, y2_doppler_calc), (255, 255, 0), 2)
                train_img_doppler = cv2.rectangle(train_img, (x1_doppler_calc, y1_doppler_calc), (x2_doppler_calc, y2_doppler_calc), (255, 255, 0), 2)
                cv2.imwrite(f'train_img_{img_idx}_with_bbox_doppler.jpg', train_img_doppler)  # @@


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
            crop_images = batch_augment(
                X, attention_map[:, :1, :, :],
                mode='crop', theta=(0.7, 0.95), padding_ratio=0.1)

        if 1:  # @@
            for idx in range(crop_images.shape[0]):
                cv2.imwrite(f'final_crop_image_{idx}.jpg', img_gpu_to_cpu(crop_images[idx]))

        # crop images forward
        y_pred_crop, _, _ = net(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(
                X, attention_map[:, 1:, :, :],
                mode='drop', theta=(0.2, 0.5))

        if 1:  # @@
            for idx in range(drop_images.shape[0]):
                cv2.imwrite(f'final_drop_image_{idx}.jpg', img_gpu_to_cpu(drop_images[idx]))

        exit(99)  # @@ !!!!!!!!

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

        writer.add_scalar("Loss/train", epoch_loss, idx)
        writer.add_scalar('Acc(Raw)/train', epoch_raw_acc[0], idx)
        writer.add_scalar('Acc(Crop)/train', epoch_crop_acc[0], idx)
        writer.add_scalar('Acc(Drop)/train', epoch_drop_acc[0], idx)

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


def validate(device, logs, validate_loader, net, pbar):

    # metrics initialization
    val_loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
      for i, (X, y, p) in enumerate(validate_loader):
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

def training(device, net, feature_center, batch_size, train_loader, doppler_train_loader, validate_loader,
             optimizer, scheduler, run_name, logs, start_epoch, total_epochs,
             savepath='.'):

    # TODO - include the 'Run/XX_d' tensorboard in output !!!!

    mc_monitor = 'val/{}'.format(raw_metric.name)
    mc = ModelCheckpoint(
        savepath=os.path.join(savepath, run_name),
        monitor=mc_monitor,
        mode='max',
        savemode_debug=True)

    if mc_monitor in logs:
        mc.set_best_score(logs[mc_monitor])
    else:
        mc.reset()

    for epoch in range(start_epoch, start_epoch + total_epochs):
        print(('#' * 10), 'epoch ', str(epoch + 1), ('#' * 10))

        mc.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, total_epochs))

        train(device, logs, train_loader, doppler_train_loader, net, feature_center, optimizer, pbar)

        validate(device, logs, validate_loader, net, pbar)

        # Checkpoints
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val/loss'])
        else:
            scheduler.step()

        mc.on_epoch_end(logs, net, feature_center=feature_center)

        #@@wandb.log(logs)
        pbar.close()
        writer.flush()

        gc.collect()
        torch.cuda.empty_cache()

    #@@wandb.finish()
    return mc.get_savepath_last()  # @@
