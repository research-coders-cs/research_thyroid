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

dir_mtrm_benign = 'Siriraj_sample_doppler_100a/Markers_Train_Remove_Markers/Benign_Remove/matched'
dir_mtrm_malignant = 'Siriraj_sample_doppler_100a/Markers_Train_Remove_Markers/Malignant_Remove/matched'
dir_dtc_benign = 'Siriraj_sample_doppler_100a/Doppler_Train_Crop/Benign/matched'
dir_dtc_malignant = 'Siriraj_sample_doppler_100a/Doppler_Train_Crop/Malignant/matched'

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
    f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0034_2_p0068.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0034_1_p0068.png',
    f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0061_2_p0122.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0061_1_p0122.png',
    f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0066_2_p0132.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0066_1_p0132.png',
    f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0090_2_p0180.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0090_1_p0180.png',
    f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0001_1_p0002.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0001_2_p0002.png',
    f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0002_1_p0005.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0002_2_p0005.png',
    f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0008_2_p0023.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0008_3_p0023.png',
    f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0013_1_p0038.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0013_3_p0038.png',
    f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0024_1_p0071.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0024_2_p0071.png',
    f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0025_1_p0074.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0025_2_p0074.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0012_1_p0035.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0012_2_p0035.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0014_2_p0041.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0014_4_p0041.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0021_1_p0062.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0021_3_p0062.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0033_1_p0008.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0033_3_p0008.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0057_1_p0080.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0057_2_p0080.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0046_1_p0047.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0046_2_p0047.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0047_1_p0050.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0047_3_p0050.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0050_1_p0059.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0050_2_p0059.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0051_1_p0062.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0051_3_p0062.png',
    f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0053_1_p0068.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0053_2_p0068.png',
    #-------- delta 'Siriraj_sample_doppler_100a' - 'Siriraj_sample_doppler_20'
    f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0107_2_p0014.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0007_2_p0014.png',
    f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0122_1_p0044.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0022_1_p0044.png',
    f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0135_1_p0070.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0035_1_p0070.png',
    f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0140_1_p0080.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0040_1_p0080.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0001_2_p0002.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0001_4_p0002.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0002_1_p0005.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0002_3_p0005.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0008_2_p0023.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0008_3_p0023.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0003_1_p0008.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0003_2_p0008.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0009_2_p0026.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0009_3_p0026.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0010_1_p0029.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0010_4_p0029.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0004_2_p0011.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0004_4_p0011.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0011_2_p0032.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0011_3_p0032.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0012_2_p0035.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0012_4_p0035.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0005_2_p0014.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0005_4_p0014.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0014_2_p0038.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0014_3_p0038.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0006_2_p0017.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0006_3_p0017.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0015_2_p0041.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0015_3_p0041.png',
    f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0007_2_p0020.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0007_4_p0020.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0004_2_p0012.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0004_3_p0013.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0132_1_p0100.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0132_2_p0101.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0133_1_p0103.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0133_2_p0104.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0134_1_p0106.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0134_2_p0107.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0136_1_p0111.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0136_2_p0112.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0137_1_p0114.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0137_2_p0115.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0022_1_p0029.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0022_2_p0030.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0138_1_p0117.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0138_2_p0118.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0023_2_p0033.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0023_3_p0034.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0139_1_p0120.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0139_2_p0121.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0024_1_p0036.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0024_2_p0037.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0140_1_p0123.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0140_1_p0124.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0025_2_p0040.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0025_3_p0041.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0141_1_p0126.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0141_2_p0127.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0026_2_p0044.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0026_3_p0045.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0142_1_p0129.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0142_2_p0130.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0027_1_p0047.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0027_2_p0048.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0143_1_p0132.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0143_2_p0133.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0028_2_p0051.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0028_3_p0052.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0029_2_p0055.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0029_3_p0056.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0030_1_p0058.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0030_2_p0059.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0146_1_p0139.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0146_2_p0140.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0031_1_p0061.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0031_2_p0062.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0147_1_p0142.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0147_2_p0143.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0035_2_p0065.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0035_3_p0066.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0148_1_p0145.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0148_2_p0146.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0149_1_p0148.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0149_2_p0149.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0150_1_p0151.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0150_2_p0152.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0066_1_p0072.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0066_2_p0073.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0151_2_p0155.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0151_2_p0156.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0152_1_p0158.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0152_2_p0159.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0153_1_p0161.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0153_2_p0162.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0125_1_p0079.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0125_2_p0080.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0154_1_p0164.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0154_2_p0165.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0155_1_p0167.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0155_2_p0168.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0127_2_p0085.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0127_3_p0086.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0156_1_p0170.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0156_2_p0171.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0128_1_p0088.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0128_2_p0089.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0158_1_p0176.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0158_2_p0177.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0129_1_p0091.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0129_2_p0092.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0159_1_p0179.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0159_2_p0180.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0160_1_p0182.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0160_2_p0183.png',
    f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0131_1_p0097.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0131_2_p0098.png',
    f'{dir_mtrm_malignant}/malignant_nodule2_c0013_1_p0003.png': f'{dir_dtc_malignant}/malignant_nodule2_c0013_3_p0003.png',
    f'{dir_mtrm_malignant}/malignant_nodule2_c0018_1_p0009.png': f'{dir_dtc_malignant}/malignant_nodule2_c0018_4_p0009.png',
    f'{dir_mtrm_malignant}/malignant_nodule2_c0017_1_p0006.png': f'{dir_dtc_malignant}/malignant_nodule2_c0017_4_p0006.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0088_1_p0017.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0088_2_p0017.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0111_2_p0063.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0111_3_p0063.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0089_1_p0019.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0089_2_p0019.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0112_2_p0065.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0112_3_p0065.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0091_2_p0023.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0091_3_p0023.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0092_1_p0025.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0092_2_p0025.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0093_1_p0027.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0093_3_p0027.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0118_2_p0077.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0118_3_p0077.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0096_1_p0033.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0096_2_p0033.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0119_1_p0079.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0119_3_p0079.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0121_1_p0083.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0121_3_p0083.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0104_2_p0049.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0104_3_p0049.png',
    f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0123_1_p0087.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0123_2_p0087.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0005_1_p0003.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0005_2_p0004.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0069_2_p0127.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0069_3_p0128.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0007_1_p0009.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0007_2_p0010.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0008_1_p0012.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0008_2_p0013.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0076_1_p0150.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0076_2_p0151.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0011_1_p0022.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0011_2_p0023.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0080_1_p0162.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0080_2_p0163.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0081_2_p0166.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0081_3_p0167.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0082_1_p0169.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0082_2_p0170.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0014_2_p0028.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0014_3_p0029.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0087_1_p0180.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0087_2_p0181.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c001x_2_p0137.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c001x_3_p0138.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0095_1_p0207.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0095_2_p0208.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0097_2_p0215.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0097_1_p0216.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0098_1_p0218.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0098_2_p0219.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0099_1_p0221.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0099_2_p0222.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0101_2_p0227.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0101_3_p0228.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0044_1_p0064.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0044_2_p0065.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0105_1_p0239.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0105_2_p0240.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0108_2_p0249.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0108_3_p0250.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0110_2_p0256.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0110_3_p0257.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0051_1_p0082.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0051_2_p0083.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0113_2_p0266.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0113_3_p0267.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0053_1_p0087.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0053_2_p0088.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0054_1_p0090.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0054_2_p0091.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0055_1_p0093.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0055_2_p0094.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0121_2_p0286.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0121_3_p0287.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0122_2_p0290.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0122_3_p0291.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0123_1_p0293.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0123_2_p0294.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0063_1_p0113.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0063_2_p0114.png',
    f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0124_1_p0296.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0124_2_p0297.png',
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
        if bbox_raw is None:
            print('@@ detect_doppler() failed for:', path_doppler)
            return bbox_to_hw_slices(bbox_crop)

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
