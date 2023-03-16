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


def get_sample_paths():
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


to_doppler = {  # TODO add 'Markers_Train_Remove_Markers' support
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
