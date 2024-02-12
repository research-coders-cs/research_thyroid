import torch
print('@@ torch.__version__:', torch.__version__)

from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

from tqdm import tqdm
#from tqdm.notebook import tqdm

import logging
logging.basicConfig(level=logging.INFO)

import matplotlib
print('@@ matplotlib.__version__:', matplotlib.__version__)

import os
import random
import time
from glob import glob


#----

def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize / 0.5), int(resize / 0.5))),#0.5
            #transforms.Resize(size=(int(resize / 0.875), int(resize / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            #@@!!!! transforms.RandomRotation(5, resample=Image.BILINEAR),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif phase == 'val':
        return transforms.Compose([
            transforms.Resize(size=(int(resize /0.875), int(resize / 0.875))),
            transforms.CenterCrop(resize),
            #transforms.RandomCrop(resize),
            #transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize), int(resize))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class ThyroidDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Cars images and labels
    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=80):
        self.phase = phase
        self.resize = resize

        if phase == 'train':
            self.images = benign_paths + benign_paths + malignant_paths + malignant_paths + malignant_paths + malignant_paths
        elif phase == 'val':
            self.images = val_benign_paths + val_malignant_paths
        elif phase == 'test':
            self.images = test_benign_paths + test_malignant_paths

        print('@@ [ThyroidDataset.__init__()] self.phase:', self.phase)
        print('@@ [ThyroidDataset.__init__()] self.resize:', self.resize)

        print('@@ [ThyroidDataset.__init__()] len(self.images):', len(self.images))
        #@@ e.g. 718 (=133*2+113*4) <- 'train'
        #@@ e.g. 47 (=29+18) <- 'val'

        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        path = self.images[item]
        image = self.transform(Image.open(path).convert('RGB'))

        if "Benign" in path:
            label = 0
        else:
            label = 1

        return image, label, path

    def __len__(self):
        return len(self.images)


#----

EPSILON = 1e-12

## Bilinear Attention Pooling

class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

## WS-DAN: Weakly Supervised Data Augmentation Network for FGVC

class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if 'inception' in net:
            self.features = models.inception_v3(pretrained=pretrained).get_features_mixed_6c()
            self.num_features = 769
        elif 'vgg' in net:
            self.features = models.vgg16(pretrained=pretrained).features
            self.num_features = 512
        elif 'resnet' in net:
            #resnet = models.resnet18(pretrained=pretrained)
            resnet = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(
              resnet.conv1,
              resnet.bn1,
              resnet.relu,
              resnet.maxpool,
              resnet.layer1,
              resnet.layer2,
              resnet.layer3,
              resnet.layer4,
            )
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)
        #self.softmax = nn.Softmax(dim=1)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)
        #p = self.softmax(p)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        return p, feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)

#----

##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)

#

# Skeleton based class
class Metric(object):
    pass

class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num

class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples

#

class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass

class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, (self.savepath+"_%.3f") % self.best_score)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, (self.savepath+"_%.3f") % self.best_score)

#

def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


#----

def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
       (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)

def test(batch_size, savepath=None):  # @@
    raw_accuracy = TopKAccuracyMetric()
    ref_accuracy = TopKAccuracyMetric()
    raw_accuracy.reset()
    ref_accuracy.reset()

    results = []

    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Test data')
        for i, (X, y, p) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            mean = X.mean((0,2,3)).view(1, 3, 1, 1)
            std = X.std((0,2,3)).view(1,3,1,1)

            MEAN = mean.cpu()#torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            STD = std.cpu()#torch.tensor([0.1, 0.1, 0.1]).view(1, 3, 1, 1)

            # WS-DAN
            y_pred_raw, _, attention_maps = net(X)

            # Augmentation with crop_mask
            crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.9, padding_ratio=0.05)

            y_pred_crop, _, _ = net(crop_image)
            y_pred = (y_pred_raw + y_pred_crop) / 2.


            # reshape attention maps
            attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
            attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

            # get heat attention maps
            heat_attention_maps = generate_heatmap(attention_maps)

            # raw_image, heat_attention, raw_attention
            raw_image = X.cpu() * STD + MEAN
            heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
            raw_attention_image = raw_image * attention_maps

            if savepath is not None:
                ToPILImage = transforms.ToPILImage()  # @@
                for batch_idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * batch_size + batch_idx)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * batch_size + batch_idx)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * batch_size + batch_idx)))

            results = (X, crop_image, y_pred, y, p, heat_attention_image)

            # Top K
            epoch_raw_acc = raw_accuracy(y_pred_raw, y)
            epoch_ref_acc = ref_accuracy(y_pred, y)

            # end of this batch
            batch_info = 'Val Acc: Raw ({:.2f}), Refine ({:.2f})'.format(
                epoch_raw_acc[0], epoch_ref_acc[0])
            pbar.update()
            pbar.set_postfix_str(batch_info)

        pbar.close()
    return results

#----

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def print_scores(results):  # @@
    pred = results[2]
    true = results[3]
    for (i, (y_hat,y)) in enumerate(zip(pred,true)):
        print("Case {}--{} Predict:{}---True:{}".format(
            i + 1, softmax(y_hat.cpu().numpy()),
            'Malignant' if torch.argmax(y_hat) == 1 else "Benign",
            'Malignant' if y == 1 else 'Benign'))

def print_auc(results, test_size, enable_plot=False):  # @@
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    # Compute ROC curve and ROC area for each class
    y_pred_b = np.zeros((test_size), dtype=float)
    y_pred_m = np.zeros((test_size), dtype=float)

    pred = results[2]
    true = results[3]

    if 1:
        print('!! test_size:', test_size)
        print('!! pred:', pred)
        print('!! true:', true)
        ##exit()

    y_m = true.detach().cpu().numpy()
    y_b = 1 - y_m

    for i, (y_hat, y) in enumerate(zip(pred, true)):
        y_pred_b[i] = float(y_hat[0])
        y_pred_m[i] = float(y_hat[1])

    fpr_b, tpr_b, _ = roc_curve(y_b, y_pred_b)
    fpr_m, tpr_m, _ = roc_curve(y_m, y_pred_m)
    roc_auc_b = auc(fpr_b, tpr_b)
    roc_auc_m = auc(fpr_m, tpr_m)

    print('@@ y_b:', y_b)
    print('@@ y_m:', y_m)
    # print('@@ fpr_b:', fpr_b)
    # print('@@ tpr_b:', tpr_b)
    # print('@@ fpr_m:', fpr_m)
    # print('@@ tpr_m:', tpr_m)
    print('@@ roc_auc_b:', roc_auc_b)
    print('@@ roc_auc_m:', roc_auc_m)

    if enable_plot:  # @@
        import matplotlib.pyplot as plt  # @@

        plt.figure()
        plt.plot(fpr_b, tpr_b, color = 'darkgreen',
                 lw = 2, label = "ROC Curve for Benign (AUC = %0.3f)" % roc_auc_b)
        plt.plot(fpr_m, tpr_m, color = 'darkred',
                 lw = 2, label = "ROC Curve for Malignant (AUC = %0.3f)" % roc_auc_m)
        plt.plot([0,1.0], [0,1.0], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc = 'lower right')
        plt.show()

#----

if __name__ == '__main__':

    # assert torch.cuda.is_available()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device("cuda")
    # torch.backends.cudnn.benchmark = True
    #====
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("@@ device:", device)

    #

    BASE_DIR = '.'
    malignant_paths = glob(os.path.join(BASE_DIR,'Dataset_train_test_val/Train/Malignant', '*.png'))
    benign_paths = glob(os.path.join(BASE_DIR, 'Dataset_train_test_val/Train/Benign', '*.png'))

    val_malignant_paths = glob(os.path.join(BASE_DIR,'Dataset_train_test_val/Val/Malignant', '*.png'))
    val_benign_paths = glob(os.path.join(BASE_DIR, 'Dataset_train_test_val/Val/Benign', '*.png'))

    test_malignant_paths = glob(os.path.join(BASE_DIR,'Dataset_train_test_val/Test/Malignant', '*.png'))
    test_benign_paths = glob(os.path.join(BASE_DIR, 'Dataset_train_test_val/Test/Benign', '*.png'))

    print('\n\n@@ ======== Checking dataset asset`')
    print('@@ len(malignant_paths):', len(malignant_paths))
    print('@@ len(benign_paths):', len(benign_paths))
    print('@@ len(val_malignant_paths):', len(val_malignant_paths))
    print('@@ len(val_benign_paths):', len(val_benign_paths))
    print('@@ len(test_malignant_paths):', len(test_malignant_paths))
    print('@@ len(test_benign_paths):', len(test_benign_paths))

    #

    target_resize = 80
    batch_size = 16
    num_classes = 2
    workers = 0  # @@ !!!!
    num_attention_maps = 16


    # cross_entropy_loss = nn.CrossEntropyLoss()
    # center_loss = CenterLoss()


    # logs = {}
    # start_epoch = 0
    # total_epochs = 160

    print('\n\n@@ ======== Calling `net = WSDAN(...)`')
    net = WSDAN(num_classes=num_classes, M=num_attention_maps, net='resnet', pretrained=True) #'vgg'

    # feature_center = torch.zeros(num_classes, num_attention_maps * net.num_features).to(device)

    net.to(device)
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)

    #

    # learning_rate = logs['lr'] if 'lr' in logs else 1e-3
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    #

    # loss_container = AverageMeter(name='loss')
    # raw_metric = TopKAccuracyMetric()
    # crop_metric = TopKAccuracyMetric()
    # drop_metric = TopKAccuracyMetric()
    #
    # callback_monitor = 'val_{}'.format(raw_metric.name)
    # callback = ModelCheckpoint(savepath=os.path.join('./', 'thyroid'),
    #     monitor=callback_monitor,
    #     mode='max')
    #
    # if callback_monitor in logs:
    #     callback.set_best_score(logs[callback_monitor])
    # else:
    #     callback.reset()

    #

    net_fname = "net_debug.pth"
    #net_fname = "resnet34_batch4_epoch100.ckpt"  # NG
    if 1:
        print(f"\n\n@@ ======== Using a saved model: {net_fname}")
        net.load_state_dict(torch.load(net_fname))
    else:
        print(f"\n\n@@ ======== Creating a new model: {net_fname}")
        training(net, target_resize, batch_size, workers)  # @@ @@
        torch.save(net.state_dict(), net_fname)
        print(f"Saved PyTorch Model State to {net_fname}")

    #

    test_dataset = ThyroidDataset(phase='test', resize=target_resize)

    test_size = len(test_dataset)
    test_loader = DataLoader(test_dataset,
        batch_size=test_size,
        shuffle=False,
        num_workers=0,  # @@ !!!!
        pin_memory=True)

    print('@@ test_size:', test_size)

    #

    results = test(test_size, savepath='./result_legacy')

    #

    print('\n\n@@ ======== print_scores(results)')
    print_scores(results)

    _enable_plot = 1  # @@
    print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
    print_auc(results, test_size, enable_plot=_enable_plot)
