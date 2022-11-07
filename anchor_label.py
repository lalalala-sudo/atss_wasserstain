# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:53:41 2018

@author: LongJun
"""
#import tensorflow as tf
import numpy as np
import tensorflow as tf
import config as cfg


def calculate_IOU (target_boxes, gt_boxes):  #gt_boxes[num_obj,4] targer_boxes[w*h*k,4]
    num_gt = gt_boxes.shape[0] 
    num_tr = target_boxes.shape[0]
    IOU_s = np.zeros((num_gt,num_tr), dtype=np.float)
    for ix in range(num_gt):
        gt_area = (gt_boxes[ix,2]-gt_boxes[ix,0]) * (gt_boxes[ix,3]-gt_boxes[ix,1])
        #print (gt_area)
        for iy in range(num_tr):
            iw = min(gt_boxes[ix,2],target_boxes[iy,2]) - max(gt_boxes[ix,0],target_boxes[iy,0])
            #print (iw)
            if iw > 0:
                ih = min(gt_boxes[ix,3],target_boxes[iy,3]) - max(gt_boxes[ix,1],target_boxes[iy,1])
                #print (ih)
                if ih > 0:
                    tar_area = (target_boxes[iy,2]-target_boxes[iy,0]) * (target_boxes[iy,3]-target_boxes[iy,1])
                    #print (tar_area)
                    i_area = iw * ih
                    iou = i_area/float((gt_area+tar_area-i_area))
                    IOU_s[ix,iy] = iou
    IOU_s = np.transpose(IOU_s)
    return IOU_s
def calculate_wasserstain(target_boxes, gt_boxes):
    num_gt = gt_boxes.shape[0]
    num_tr = target_boxes.shape[0]
    #IOU_s = np.zeros((num_gt, num_tr), dtype=np.float)
    center1 = (target_boxes[..., :, None, :2] + target_boxes[..., :, None, 2:]) / 2
    center2 = (gt_boxes[..., None, :, :2] + gt_boxes[..., None, :, 2:]) / 2
    whs = center1[..., :2] - center2[..., :2]
    eps=cfg.eps
    constant=cfg.constant

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

    w1 = target_boxes[..., :, None, 2] - target_boxes[..., :, None, 0] + eps
    h1 = target_boxes[..., :, None, 3] - target_boxes[..., :, None, 1] + eps
    w2 = gt_boxes[..., None, :, 2] - gt_boxes[..., None, :, 0] + eps
    h2 = gt_boxes[..., None, :, 3] - gt_boxes[..., None, :, 1] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wassersteins = np.sqrt(center_distance + wh_distance)

    normalized_wasserstein = np.exp(-wassersteins / constant)

    return normalized_wasserstein


def lables_generate_atss(gt_boxes, target_boxes, overlaps_pos, overlaps_neg, im_width, im_height,flag="iou"):
    total_targets = target_boxes.shape[0]
    targets_inside = np.where((target_boxes[:, 0] > 0) & \
                              (target_boxes[:, 2] < im_width) & \
                              (target_boxes[:, 1] > 0) & \
                              (target_boxes[:, 3] < im_height))[0]
    targets = target_boxes[targets_inside]
    labels = np.empty((targets.shape[0],), dtype=np.float32)
    labels.fill(-1)
    if flag == "iou":
        IOUs = calculate_IOU(targets, gt_boxes)  # IOUs:anchor*gt
    else:
        IOUs = calculate_wasserstain(targets, gt_boxes)
    #  计算真实框的中心点
    gt_cx=(gt_boxes[:,2]+gt_boxes[:,0])/2.0
    gt_cy = (gt_boxes[:, 3] + gt_boxes[:,1]) / 2.0
    gt_point=np.stack(gt_cx,gt_cy,axis=1)

    # 计算锚框的中心点
    anchor_cx=(targets[:,0]+targets[:,2])/2.0
    anchor_cy = (targets[:, 1] + targets[:, 3]) / 2.0
    anchor_point=np.stack(anchor_cx,anchor_cy,axis=1)

    s1 = np.power((anchor_point[:, None, :] - gt_point[None, :, :]), 2)
    s2 = np.sum(s1, axis=-1)
    distance = np.sqrt(s2)
    distance = np.transpose(distance)
    topk = cfg.topk
    # gt*topk

    top_ids = tf.nn.top_k(-distance, topk).indices
    with tf.Session() as sess:
        top_id = top_ids.eval()

    cols = np.empty((gt_box.shape[0], topk), dtype=int)
    for i in range(gt_box.shape[0]):
        cols[i] = i

    candidate_iou = IOUs[top_id, cols]

    # 计算出阈值
    iou_mean = np.mean(candidate_iou, axis=1)
    iou_std = np.transpose(np.std(candidate_iou, axis=1))
    iou_thresh = iou_mean + iou_std

    # 计算出正样本的id
    is_pos = np.transpose(candidate_iou) >= iou_thresh

    top_ids = np.transpose(top_id)
    top_ids = top_ids.flatten()

    is_pos = is_pos.flatten()

    index = top_ids[is_pos]
    neg_id = np.setdiff1d(targets_inside, index)
    labels[index] = 1
    labels[neg_id] = 0

    anchor_obj = np.argmax(IOUs,axis=1)  # anchor_obj is the index of a grounturth which has the highest IOU with this anchor
    labels = fill_label(labels, total_targets, targets_inside)  # set the outside anchor with label -1
    anchor_obj = fill_label(anchor_obj, total_targets, targets_inside, fill=-1)
    anchor_obj = anchor_obj.astype(np.int64)
    return labels,anchor_obj



def labels_generate (gt_boxes, target_boxes, overlaps_pos, overlaps_neg, im_width, im_height,flag="iou"):
    """ generate the anchor labels, -1 means unusefl anchor, 1 means positive anchor, 0 means negative anchor"""
    total_targets = target_boxes.shape[0]
    targets_inside = np.where((target_boxes[:,0]>0)&\
                              (target_boxes[:,2]<im_width)&\
                              (target_boxes[:,1]>0)&\
                              (target_boxes[:,3]<im_height))[0]
    targets = target_boxes[targets_inside]
    labels = np.empty((targets.shape[0],), dtype=np.float32)
    labels.fill(-1)   # lables_shape:1*targets.shape[0]
    if flag=="iou":
        IOUs = calculate_IOU(targets, gt_boxes) # IOUs:anchor*gt
    else:
        IOUs=calculate_wasserstain(targets,gt_boxes)
    # 返回每一个anchor最接近的gt下标 1*targets
    max_gt_arg = np.argmax(IOUs, axis=1)
    # 返回每一个anchor与最接近的gt之间的IOU值 1*targets
    max_IOUS = IOUs[np.arange(len(targets_inside)), max_gt_arg]
    # 如果最大的ious小于最小值，则赋值0
    labels[max_IOUS < overlaps_neg] = 0
    # 返回每一个gt最接近的anchor的下标 1*gt
    max_anchor_arg = np.argmax(IOUs, axis=0)
    # 与gt的iou是最大值的target赋值1
    labels[max_anchor_arg] = 1
    # 如果最大的ious大于最大值，则赋值1
    labels[max_IOUS > overlaps_pos] = 1
    anchor_obj = max_gt_arg #anchor_obj is the index of a grounturth which has the highest IOU with this anchor
    labels = fill_label(labels, total_targets, targets_inside) #set the outside anchor with label -1
    anchor_obj = fill_label(anchor_obj, total_targets, targets_inside, fill=-1)
    anchor_obj = anchor_obj.astype(np.int64)
    return labels, anchor_obj



def labels_filt (labels, anchor_batch):
    """ label filt: get 256 anchor where 50% for positive anchor, 50 for negative anchor"""
    max_fg_num = anchor_batch*0.5
    fg_inds = np.where(labels==1)[0]
    if len(fg_inds) > max_fg_num:
        disable_inds = np.random.choice(fg_inds, size=int(len(fg_inds) - max_fg_num), replace=False)
        labels[disable_inds] = -1
    max_bg_num = anchor_batch - np.sum(labels==1)
    bg_inds = np.where(labels==0)[0]
    if len(bg_inds) > max_bg_num:
        disable_inds = np.random.choice(bg_inds, size=int(len(bg_inds) - max_bg_num), replace=False)
        labels[disable_inds] = -1
    return labels

def anchor_labels_process(boxes, conners, anchor_batch, overslaps_max, overslaps_min, im_width, im_height,flag,atss):
    if atss=="atss":
        labels, anchor_obj = labels_generate(boxes, conners, overslaps_max, overslaps_min, im_width, im_height,flag)
    else:
        labels, anchor_obj = lables_generate_atss(boxes, conners, overslaps_max, overslaps_min, im_width, im_height, flag)
    labels = labels_filt(labels, anchor_batch)
    return labels, anchor_obj

def fill_label(labels, total_target, target_inside, fill=-1):
    new_labels = np.empty((total_target, ), dtype=np.float32)
    new_labels.fill(fill)
    new_labels[target_inside] = labels
    return new_labels

if __name__ == '__main__':
    anchor=np.arange(24).reshape((6, 4))
    gt_box=np.arange(8).reshape((2,4))
    IOUs = calculate_IOU(anchor, gt_box)
    print(IOUs)
    """
    total_targets = anchor.shape[0]
    targets_inside = np.where((anchor[:, 0] > -1) & \
                              (anchor[:, 2] < 100) & \
                              (anchor[:, 1] > -1) & \
                              (anchor[:, 3] < 100))[0]
    targets = anchor[targets_inside]
    labels = np.empty((targets.shape[0],), dtype=np.float32)
    labels.fill(-1)
    IOUs = calculate_IOU(targets, gt_box)
    #  计算真实框的中心点
    gt_cx = (gt_box[:, 2] + gt_box[:, 0]) / 2.0
    gt_cy = (gt_box[:, 3] + gt_box[:, 1]) / 2.0
    gt_point = np.stack((gt_cx, gt_cy),axis=1)

    # 计算锚框的中心点
    anchor_cx = (targets[:, 0] + targets[:, 2]) / 2.0
    anchor_cy = (targets[:, 1] + targets[:, 3]) / 2.0
    anchor_point = np.stack((anchor_cx, anchor_cy), axis=1)

    s1 = np.power((anchor_point[:, None, :] - gt_point[None, :, :]),2)
    s2=np.sum(s1,axis=-1)
    distance=np.sqrt(s2)
    distance=np.transpose(distance)
    topk=3
    # gt*topk

    top_ids=tf.nn.top_k(-distance,topk).indices
    with tf.Session() as sess:
        top_id=top_ids.eval()

    cols = np.empty((gt_box.shape[0],topk),dtype=int)
    for i in range(gt_box.shape[0]):
        cols[i] = i

    candidate_iou=IOUs[top_id,cols]

    # 计算出阈值
    iou_mean=np.mean(candidate_iou,axis=1)
    iou_std=np.transpose(np.std(candidate_iou,axis=1))
    iou_thresh=iou_mean+iou_std

    # 计算出正样本的id
    is_pos=np.transpose(candidate_iou)>=iou_thresh

    top_ids=np.transpose(top_id)
    top_ids=top_ids.flatten()

    is_pos=is_pos.flatten()

    index=top_ids[is_pos]
    neg_id=np.setdiff1d(targets_inside, index)
    labels[index]=1
    labels[neg_id]=0



    anchor_obj = np.argmax(IOUs, axis=1)  # anchor_obj is the index of a grounturth which has the highest IOU with this anchor
    labels = fill_label(labels, total_targets, targets_inside)  # set the outside anchor with label -1
    anchor_obj = fill_label(anchor_obj, total_targets, targets_inside, fill=-1)
    anchor_obj = anchor_obj.astype(np.int64)
    """







                        
    
