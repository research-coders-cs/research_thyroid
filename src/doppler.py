import os
import cv2
import numpy as np
import pandas as pd


def detect_doppler(img):
    if len(img.shape) < 3:
        #print('gray_scale')
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    green_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    green_image[:,:] = img[:,:,1]
    _, threshold = cv2.threshold(green_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite('check.jpg', threshold)

    cntrRect = None
    maxarea = 0
    for cnt in contours:
        hull = cv2.convexHull(cnt,returnPoints = True)
        #print(cnt)
        #print(hull)
        approx = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(cnt, True), True)
        #cv2.drawContours(img, [approx], 0, (0), 5)
        if len(approx) == 4:
            xy = approx.ravel()
            area = cv2.contourArea(cnt)
            if maxarea < area:
                use = True
                for i in range(4):
                    lx1 = xy[i*2]
                    ly1 = xy[i*2+1]
                    if i == 3:
                        lx2 = xy[0]
                        ly2 = xy[1]
                    else:
                        lx2 = xy[(i+1)*2]
                        ly2 = xy[(i+1)*2+1]
                    angle = np.absolute(np.arctan2(ly2-ly1, lx2-lx1)) * 180 / np.pi
                    if angle > 45 and angle < 135:
                        angle = angle - 90
                    if angle >= 135:
                        angle = angle - 180
                    if angle > 3:
                        use = False
                        break

                if use:
                    maxarea = area
                    cntrRect = approx

    if cntrRect is not None:
        xy = cntrRect.ravel()
        x = [xy[0], xy[2], xy[4], xy[6]]
        y = [xy[1], xy[3], xy[5], xy[7]]
        min_x = float(min(x))
        max_x = float(max(x))
        min_y = float(min(y))
        max_y = float(max(y))
        x = [min_x, min_y, max_x, max_y]
        return x
    return None

def get_iou(truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(truth[0], pred[0])
    iy1 = np.maximum(truth[1], pred[1])
    ix2 = np.minimum(truth[2], pred[2])
    iy2 = np.minimum(truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width
    # print("area_of_intersection : ", area_of_intersection)

    w_mark =  (pred[2] - pred[0])
    h_mark = (pred[3] - pred[1])
    area_mark = w_mark * h_mark
    # print("area_mark : ", area_mark)

    # Ground Truth dimensions.
    gt_height = truth[3] - truth[1] + 1
    gt_width = truth[2] - truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    # print("area_of_union : ", area_of_union)

    _iou = area_of_intersection / area_of_union
    #@@ iou = np.round(_iou * 100, 2)

    _intersection_of_mark = area_of_intersection / area_mark
    #@@ intersection_of_mark = np.round(_intersection_of_mark * 100, 2)

    print('@@ get_iou(): area_{of_intersection,mark,of_union}, intersection_of_mark: %0.1f, %0.1f, %0.1f' % (
        area_of_intersection, area_mark, area_of_union))

    #@@return iou, intersection_of_mark
    return _iou, _intersection_of_mark

#

def doppler_comp(path_doppler, path_markers, path_markers_label):
    img_doppler = cv2.imread(path_doppler)
    width = int(img_doppler.shape[1])
    height = int(img_doppler.shape[0])

    temp = detect_doppler(img_doppler)

    x1_doppler_calc = int(temp[0])
    y1_doppler_calc = int(temp[1])
    x2_doppler_calc = int(temp[2])
    y2_doppler_calc = int(temp[3])

    bbox_doppler = np.array([
        x1_doppler_calc, y1_doppler_calc, x2_doppler_calc, y2_doppler_calc],
        dtype=np.float32)
    border_img_doppler = cv2.rectangle(img_doppler, (x1_doppler_calc, y1_doppler_calc), (x2_doppler_calc, y2_doppler_calc), (255, 255, 0), 2)

    #

    label_markers = pd.read_csv(path_markers_label, header=None, index_col=False).to_numpy()
    temp_row = []
    for row in label_markers:
        temp_col = []
        for col in row:
            for item in col.split():
                temp_col.append(float(item))
        temp_row.append(np.array(temp_col))
    temp = np.array(temp_row)

    x1_markers = np.min(temp[:,1])
    x2_markers = np.max(temp[:,1])
    y1_markers = np.min(temp[:,2])
    y2_markers = np.max(temp[:,2])

    x1_markers_calc = int(width * x1_markers)
    y1_markers_calc = int(height * y1_markers)
    x2_markers_calc = int(width * x2_markers)
    y2_markers_calc = int(height * y2_markers)
    bbox_markers = np.array(
        [x1_markers_calc, y1_markers_calc, x2_markers_calc, y2_markers_calc],
        dtype=np.float32)

    #

    img_markers = cv2.imread(path_markers)

    unborder_img_markers = cv2.resize(img_markers, (width, height))
    unborder_img_markers = cv2.rectangle(
        unborder_img_markers,
        (int(width * x1_doppler_calc), int(height * y1_doppler_calc)),
        (int(width * x2_doppler_calc), int(height * y2_doppler_calc)),
        (255, 255, 0), 2)
    border_img_markers = cv2.rectangle(
        unborder_img_markers.copy(),
        (int(width * x1_markers), int(height * y1_markers)),
        (int(width * x2_markers), int(height * y2_markers)),
        (255, 0, 0), 2)

    # doppler
    border_img_markers = cv2.rectangle(
        border_img_markers,
        (x1_doppler_calc, y1_doppler_calc),
        (x2_doppler_calc, y2_doppler_calc),
        (255, 255, 0), 2)

    return bbox_doppler, bbox_markers, border_img_doppler, border_img_markers


def plot_comp(border_img_doppler, border_img_markers, path_doppler, path_markers):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(20, 15))
    for idx, (title, border_img) in enumerate((
            (f'Doppler : {os.path.basename(path_doppler)}', border_img_doppler),
            (f'Markers : {os.path.basename(path_markers)}', border_img_markers))):
        ax[idx].title.set_text(title)
        ax[idx].imshow(border_img)
        ax[idx].grid(False)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])

    return plt


def get_sample_paths():  # used by `demo_doppler_compare()`
    return (
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0011_1_p0022.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0011_2_p0022.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0011_2_p0022.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0076_1_p0152.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0076_2_p0152.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0076_2_p0152.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0022_1_p0044.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0022_2_p0044.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0022_2_p0044.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule3_0001-0030_c0024_2_p0071.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule3_0001-0030_c0024_1_p0071.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule3_0001-0030_c0024_1_p0071.txt'),
        # !! removed jpg causing issues on data_loader
        # ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule2_0001-0016_c0001_3_p0002.jpg',
        # './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule2_0001-0016_c0001_1_p0002.png',
        # './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule2_0001-0016_c0001_1_p0002.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_siriraj_0001-0160_c0128_2_p0089.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_siriraj_0001-0160_c0128_1_p0088.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_siriraj_0001-0160_c0128_1_p0088.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0008_1_p0016.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0008_2_p0016.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0008_2_p0016.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_siriraj_0001-0124_c0110_3_p0257.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_nodule3_0001-0030_c0004_3_p0011.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.txt'),
    )

#

dir_20_mtrm_benign = 'Siriraj_sample_doppler_20/Markers_Train_Remove_Markers/Benign_Remove/matched'
dir_20_mtrm_malignant = 'Siriraj_sample_doppler_20/Markers_Train_Remove_Markers/Malignant_Remove/matched'
dir_20_dtc_benign = 'Siriraj_sample_doppler_20/Doppler_Train_Crop/Benign/matched'
dir_20_dtc_malignant = 'Siriraj_sample_doppler_20/Doppler_Train_Crop/Malignant/matched'

to_doppler = {
    #-------- 'Siriraj_sample_doppler_comp'
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
    #-------- 'Siriraj_sample_doppler_20'
    f'{dir_20_mtrm_benign}/benign_nodule1_0001-0100_c0034_2_p0068.png': f'{dir_20_dtc_benign}/benign_nodule1_0001-0100_c0034_1_p0068.png',
    f'{dir_20_mtrm_benign}/benign_nodule1_0001-0100_c0061_2_p0122.png': f'{dir_20_dtc_benign}/benign_nodule1_0001-0100_c0061_1_p0122.png',
    f'{dir_20_mtrm_benign}/benign_nodule1_0001-0100_c0066_2_p0132.png': f'{dir_20_dtc_benign}/benign_nodule1_0001-0100_c0066_1_p0132.png',
    f'{dir_20_mtrm_benign}/benign_nodule1_0001-0100_c0090_2_p0180.png': f'{dir_20_dtc_benign}/benign_nodule1_0001-0100_c0090_1_p0180.png',
    f'{dir_20_mtrm_benign}/benign_nodule3_0001-0030_c0001_1_p0002.png': f'{dir_20_dtc_benign}/benign_nodule3_0001-0030_c0001_2_p0002.png',
    f'{dir_20_mtrm_benign}/benign_nodule3_0001-0030_c0002_1_p0005.png': f'{dir_20_dtc_benign}/benign_nodule3_0001-0030_c0002_2_p0005.png',
    f'{dir_20_mtrm_benign}/benign_nodule3_0001-0030_c0008_2_p0023.png': f'{dir_20_dtc_benign}/benign_nodule3_0001-0030_c0008_3_p0023.png',
    f'{dir_20_mtrm_benign}/benign_nodule3_0001-0030_c0013_1_p0038.png': f'{dir_20_dtc_benign}/benign_nodule3_0001-0030_c0013_3_p0038.png',
    f'{dir_20_mtrm_benign}/benign_nodule3_0001-0030_c0024_1_p0071.png': f'{dir_20_dtc_benign}/benign_nodule3_0001-0030_c0024_2_p0071.png',
    f'{dir_20_mtrm_benign}/benign_nodule3_0001-0030_c0025_1_p0074.png': f'{dir_20_dtc_benign}/benign_nodule3_0001-0030_c0025_2_p0074.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0001-0030_c0012_1_p0035.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0001-0030_c0012_2_p0035.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0001-0030_c0014_2_p0041.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0001-0030_c0014_4_p0041.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0001-0030_c0021_1_p0062.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0001-0030_c0021_3_p0062.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0031-0060_c0033_1_p0008.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0031-0060_c0033_3_p0008.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0031-0060_c0039_2_p0026.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0031-0060_c0039_3_p0026.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0031-0060_c0046_1_p0047.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0031-0060_c0046_2_p0047.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0031-0060_c0047_1_p0050.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0031-0060_c0047_3_p0050.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0031-0060_c0050_1_p0059.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0031-0060_c0050_2_p0059.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0031-0060_c0051_1_p0062.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0031-0060_c0051_3_p0062.png',
    f'{dir_20_mtrm_malignant}/malignant_nodule3_0031-0060_c0053_1_p0068.png': f'{dir_20_dtc_malignant}/malignant_nodule3_0031-0060_c0053_2_p0068.png',
    #--------
}

#

def bbox_to_hw_slices(bbox):
    return (
        slice(int(bbox[1]), int(bbox[3])),  # i.e. height_min:height_max
        slice(int(bbox[0]), int(bbox[2])))  # i.e. width_min:width_max

def bbox_draw(img, bbox, color=(255, 0, 0), thickness=1):
    return cv2.rectangle(img,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])), color, thickness)

# ======== TODO refactor ^^ into preprocessing part i.e. `ThyroidDataset()`
def resolve_hw_slices(bbox_crop, train_img_copy, train_img_path, idx, size, savepath):
    THRESH_ISEC_IN_CROP = 0.25

    path_doppler = to_doppler[train_img_path] if train_img_path in to_doppler else None
    if path_doppler is not None:  # @@
        # get doppler bbox (scaled)
        raw = cv2.imread(path_doppler)
        bbox_raw = detect_doppler(raw)
        bbox = np.array([
            bbox_raw[0] * size[0] / raw.shape[1], bbox_raw[1] * size[1] / raw.shape[0],
            bbox_raw[2] * size[0] / raw.shape[1], bbox_raw[3] * size[1] / raw.shape[0]],
            dtype=np.float32)

        iou, isec_in_crop = get_iou(bbox, bbox_crop)
        print('@@ THRESH_ISEC_IN_CROP:', THRESH_ISEC_IN_CROP)
        qualify = 1 if iou > 1e-4 and isec_in_crop > THRESH_ISEC_IN_CROP else 0
        debug_fname_jpg = f'debug_crop_doppler_{idx}_iou_%0.4f_isecincrop_%0.3f_qualify_%d.jpg' % (
            iou, isec_in_crop, qualify)
        print('@@ debug_fname_jpg:', debug_fname_jpg)

        if 1:  # debug dump
            bbox_draw(train_img_copy, bbox, (255, 255, 0), 1)  # blue
            bbox_draw(train_img_copy, bbox_crop, (0, 0, 255), 1)  # red
            cv2.imwrite(os.path.join(savepath, debug_fname_jpg), train_img_copy)

            # crop patch image; OK
            sh_, sw_ = bbox_to_hw_slices(bbox_crop)
            img_ = train_img_copy.copy()[sh_, sw_, :]
            cv2.imwrite(os.path.join(savepath, f'debug_crop_idx_{idx}.jpg'), img_)

            # doppler patch image; OK
            sh_, sw_ = bbox_to_hw_slices(bbox)
            img_ = train_img_copy.copy()[sh_, sw_, :]
            cv2.imwrite(os.path.join(savepath, f'debug_doppler_idx_{idx}.jpg'), img_)

    return bbox_to_hw_slices(bbox_crop if qualify else bbox)
