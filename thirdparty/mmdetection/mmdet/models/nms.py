from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from mmengine.structures import InstanceData
from torchvision.ops import nms


def soft_nms_pytorch(dets, scores, iou_threshold=0.3, sigma=0.5, method='gaussian', min_score=0.001):
    # dets: Tensor[N, 4], scores: Tensor[N]
    N = dets.shape[0]
    indices = torch.arange(N).to(dets.device)
    for i in range(N):
        # Find the detection with the highest score
        maxpos = i + torch.argmax(scores[i:], dim=0).item()
        if scores[maxpos] > scores[i]:
            # Swap detections
            dets[i], dets[maxpos] = dets[maxpos].clone(), dets[i].clone()
            scores[i], scores[maxpos] = scores[maxpos].clone(), scores[i].clone()
            indices[i], indices[maxpos] = indices[maxpos], indices[i]
        
        # Compute IoU of the highest-score detection with the rest
        pos = i + 1
        if pos >= N:
            break
        xx1 = torch.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = torch.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = torch.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = torch.minimum(dets[i, 3], dets[pos:, 3])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        areas_i = (dets[i, 2] - dets[i, 0]) * (dets[i, 3] - dets[i, 1])
        areas = (dets[pos:, 2] - dets[pos:, 0]) * (dets[pos:, 3] - dets[pos:, 1])
        union = areas_i + areas - inter
        ious = inter / union

        # Apply score decay
        if method == 'linear':
            weight = torch.ones_like(ious)
            mask = ious > iou_threshold
            weight[mask] = weight[mask] - ious[mask]
        elif method == 'gaussian':
            weight = torch.exp(-((ious ** 2) / sigma))
        else:  # 'naive'
            weight = torch.ones_like(ious)
            weight[ious > iou_threshold] = 0.0

        scores[pos:] *= weight

    # Filter out detections with scores below min_score
    keep = scores > min_score
    return dets[keep], scores[keep], indices[keep]


def wrist_keypoint_nms(results, img_h, img_w, wrist_distance_threshold: float = 1):
    """
    Perform classwise NMS based solely on the wrist keypoint distance.
    
    For each detection class, the function keeps the highest scored detection and
    suppresses any others for which the Euclidean distance between their wrist keypoints
    is below the given threshold (i.e. about 2-3 pixels apart).
    
    Assumptions:
      - results.bboxes: Tensor of shape (N, 4)
      - results.scores: Tensor of shape (N,)
      - results.keypoints: Tensor of shape (N, 3, 2) 
          (three keypoints per detection; 2D coordinates)
      - results.labels: Tensor of shape (N,), containing integer labels.
      
    The mapping from label to wrist keypoint index is:
      - For forearm classes (labels 0 and 1): use wrist at keypoint index 2.
      - For hand classes (labels 2 and 3): use wrist at keypoint index 0.
      
    :param results: InstanceData with fields bboxes, scores, keypoints, labels.
    :param wrist_distance_threshold: float threshold in pixels (default ~3.0)
    :return: InstanceData containing the kept detections.
    """
    # Extract data
    bboxes = results.bboxes           # Tensor of shape (N, 4)
    scores = results.scores           # Tensor of shape (N,)
    keypoints = results.keypoints     # Tensor of shape (N, 3, 2)
    labels = results.labels           # Tensor of shape (N,)
    device = bboxes.device

    img_size = torch.tensor([img_w, img_h], device=device, dtype=torch.float32)

    # Ensure tensors are on the same device
    scores = scores.to(device)
    keypoints = keypoints.to(device)
    labels = labels.to(device)

    # Define mapping from class label to wrist keypoint index.
    # For forearms (classes 0 and 1) use keypoint index 2; for hands (classes 2 and 3) use index 0.
    wrist_idx_mapping = {0: 2, 1: 2, 2: 0, 3: 0}

    final_keep_indices = []

    # Process detections for each class separately.
    unique_labels = labels.unique()
    for cls in unique_labels:
        # Get the indices for detections of this class.
        cls_mask = (labels == cls)
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).view(-1)
        if cls_indices.numel() == 0:
            continue

        # Sort these detections by descending score.
        cls_scores = scores[cls_indices]
        sorted_scores, order = torch.sort(cls_scores, descending=True)
        sorted_cls_indices = cls_indices[order]

        # Get the appropriate wrist keypoint index for this class.
        wrist_idx = wrist_idx_mapping[int(cls.item())]

        # Perform greedy NMS based on wrist keypoint distance.
        # (Keep the highest scored detection, then remove any detections
        # whose wrist keypoint is too close.)
        keep_for_cls = []
        while sorted_cls_indices.numel() > 0:
            current = sorted_cls_indices[0]
            keep_for_cls.append(current.item())
            if sorted_cls_indices.numel() == 1:
                break

            # Get the wrist coordinate for the current detection.
            current_wrist = keypoints[current, wrist_idx, :].unsqueeze(0)  # shape: (1, 2)
            current_wrist = current_wrist / img_size  # Normalize to [0, 1]
            # Get the wrist coordinates for the remaining detections.
            remaining_indices = sorted_cls_indices[1:]
            remaining_wrists = keypoints[remaining_indices, wrist_idx, :]  # shape: (M, 2)
            remaining_wrists = remaining_wrists/ img_size  # Normalize to [0, 1]

            # Compute Euclidean distances.
            distances = torch.norm(remaining_wrists - current_wrist, dim=1)
            # Keep only detections where the wrist is farther than the threshold.
            keep_mask = distances > wrist_distance_threshold

            # Update the list of indices to process.
            sorted_cls_indices = remaining_indices[keep_mask]

        final_keep_indices.extend(keep_for_cls)

    if len(final_keep_indices) == 0:
        # Return an empty results object if nothing remains.
        return type(results)()

    # Optionally, sort final indices by descending score.
    final_keep_indices = torch.tensor(final_keep_indices, device=device, dtype=torch.long)
    final_scores = scores[final_keep_indices]
    final_order = torch.argsort(final_scores, descending=True)
    final_keep_indices = final_keep_indices[final_order]


    # Prepare the final InstanceData (or similar data structure).
    bboxes = bboxes[final_keep_indices]
    scores = scores[final_keep_indices]
    labels = labels[final_keep_indices]
    keypoints = keypoints[final_keep_indices]

    iou_threshold = 0.8
    keep_indices = nms(bboxes, scores, iou_threshold)
    # Gather the detections after NMS
    nms_results = type(results)()  # Create a new instance of the same type
    nms_results.bboxes = bboxes[keep_indices]
    nms_results.scores = scores[keep_indices]
    nms_results.labels = labels[keep_indices]
    nms_results.keypoints = keypoints[keep_indices]

    return nms_results


def keypoint_nms(results: InstanceData,
                 iou_threshold: float = 0.3,
                 keypoint_threshold: float = 5.0) -> InstanceData:
    # Extract data from results
    bboxes = results.bboxes  # Tensor of shape (N, 4)
    scores = results.scores  # Tensor of shape (N,)
    keypoints = results.keypoints  # Tensor of shape (N, K, 2)
    labels = results.labels  # Tensor of shape (N,)
    device = bboxes.device

    # Ensure all tensors are on the same device
    scores = scores.to(device)
    keypoints = keypoints.to(device)
    labels = labels.to(device)

    keep_indices = nms(bboxes, scores, iou_threshold)
    # Gather the detections after NMS
    nms_bboxes = bboxes[keep_indices]
    nms_scores = scores[keep_indices]
    nms_labels = labels[keep_indices]
    nms_keypoints = keypoints[keep_indices]
        
    # If no detections remain, return empty results
    if nms_bboxes.numel() == 0:
        return InstanceData()

    # Step 2: Enhanced Keypoint Proximity Suppression
    N = nms_keypoints.shape[0]
    K = nms_keypoints.shape[1]  # Number of keypoints

    # Sort detections by descending confidence scores
    sorted_scores, sorted_indices = torch.sort(nms_scores, descending=True)
    nms_bboxes = nms_bboxes[sorted_indices]
    nms_labels = nms_labels[sorted_indices]
    nms_keypoints = nms_keypoints[sorted_indices]

    # Compute pairwise distances for each keypoint
    # nms_keypoints shape: (N, K, 2)
    keypoint_distances_per_kp = torch.stack([
        torch.cdist(nms_keypoints[:, k, :], nms_keypoints[:, k, :], p=2)
        for k in range(K)
    ])  # Shape: (K, N, N)

    # Create a mask where all keypoints are within the threshold
    keypoint_close_mask = keypoint_distances_per_kp < keypoint_threshold  # Shape: (K, N, N)
    # Now, we need to check where all keypoints are close
    all_keypoints_close = keypoint_close_mask.all(dim=0)  # Shape: (N, N)

    # Exclude self-comparisons
    all_keypoints_close.fill_diagonal_(False)

    # Create a mask to consider only detections with lower scores
    idx_i = torch.arange(N).unsqueeze(1).expand(N, N).to(device)
    idx_j = torch.arange(N).unsqueeze(0).expand(N, N).to(device)
    # Since detections are sorted by descending scores, idx_j > idx_i corresponds to lower scores
    lower_score_mask = idx_j > idx_i

    # Final suppression mask
    suppression_mask = all_keypoints_close & lower_score_mask

    # Determine which detections to suppress
    suppressed = suppression_mask.any(dim=0)

    # Indices of detections to keep after keypoint suppression
    keep_keypoints_indices = torch.nonzero(~suppressed, as_tuple=False).view(-1)

    # Update detections after keypoint suppression
    final_bboxes = nms_bboxes[keep_keypoints_indices]
    final_scores = sorted_scores[keep_keypoints_indices]
    final_labels = nms_labels[keep_keypoints_indices]
    final_keypoints = nms_keypoints[keep_keypoints_indices]

    # Step 3: Suppress Inner Bounding Boxes
    N_final = final_bboxes.shape[0]
    
    if N_final == 0:
        return InstanceData()

    x1 = final_bboxes[:, 0].unsqueeze(1)  # Shape: (N_final, 1)
    y1 = final_bboxes[:, 1].unsqueeze(1)
    x2 = final_bboxes[:, 2].unsqueeze(1)
    y2 = final_bboxes[:, 3].unsqueeze(1)

    x1_j = final_bboxes[:, 0].unsqueeze(0)  # Shape: (1, N_final)
    y1_j = final_bboxes[:, 1].unsqueeze(0)
    x2_j = final_bboxes[:, 2].unsqueeze(0)
    y2_j = final_bboxes[:, 3].unsqueeze(0)

    # Check if box_i is entirely inside box_j
    x1_in = x1 >= x1_j  # Shape: (N_final, N_final)
    y1_in = y1 >= y1_j
    x2_in = x2 <= x2_j
    y2_in = y2 <= y2_j

    # Combine conditions
    inside_mask = x1_in & y1_in & x2_in & y2_in
    inside_mask.fill_diagonal_(False)  # Exclude self-comparisons

    # Create new indices for inner box suppression to avoid variable conflicts
    idx_i_inner = torch.arange(N_final).unsqueeze(1).expand(N_final, N_final).to(device)
    idx_j_inner = torch.arange(N_final).unsqueeze(0).expand(N_final, N_final).to(device)
    # Since detections are sorted by descending scores, idx_j_inner > idx_i_inner corresponds to lower scores
    upper_triangle_inner = idx_j_inner > idx_i_inner

    # Final suppression mask for inner boxes
    inner_box_suppression_mask = inside_mask & upper_triangle_inner

    # Determine which detections to suppress based on inner boxes
    inner_box_suppressed = inner_box_suppression_mask.any(dim=0)

    # Indices of detections to keep after inner box suppression
    keep_inner_box_indices = torch.nonzero(~inner_box_suppressed, as_tuple=False).view(-1)

    # Update final detections
    final_bboxes = final_bboxes[keep_inner_box_indices]
    final_scores = final_scores[keep_inner_box_indices]
    final_labels = final_labels[keep_inner_box_indices]
    final_keypoints = final_keypoints[keep_inner_box_indices]

    # Prepare final results
    nms_results = InstanceData()
    nms_results.bboxes = final_bboxes
    nms_results.scores = final_scores
    nms_results.labels = final_labels
    nms_results.keypoints = final_keypoints

    return nms_results


def keypoint_nms2(results: InstanceData) -> InstanceData:
    # Extract data from results
    bboxes = results.bboxes  # Tensor of shape (N, 4)
    scores = results.scores  # Tensor of shape (N,)
    keypoints = results.keypoints  # Tensor of shape (N, K, 2)
    labels = results.labels  # Tensor of shape (N,)

    device = bboxes.device

    # Define thresholds
    iou_threshold = 0.3  # Adjust as needed
    keypoint_threshold = 5.0  # Adjust based on the scale of your keypoint coordinates

    # Step 1: Perform Class-Agnostic NMS
    keep_indices = nms(bboxes, scores, iou_threshold)

    # Gather the detections after NMS
    nms_bboxes = bboxes[keep_indices]
    nms_scores = scores[keep_indices]
    nms_labels = labels[keep_indices]
    nms_keypoints = keypoints[keep_indices]
        
    # Check if there are any detections left after NMS
    if nms_bboxes.numel() == 0:
        return results[keep_indices]  # Return empty results

    # Step 2: Suppress Detections Based on Keypoint Proximity
    N = nms_keypoints.shape[0]
    keypoints_flat = nms_keypoints.view(N, -1)  # Shape: (N, K*2)

    # Sort detections by descending confidence scores
    sorted_scores, sorted_indices = torch.sort(nms_scores, descending=True)
    keypoints_flat = keypoints_flat[sorted_indices]
    nms_bboxes = nms_bboxes[sorted_indices]
    nms_labels = nms_labels[sorted_indices]
    nms_keypoints = nms_keypoints[sorted_indices]

    # Compute pairwise distances between keypoints
    diff = keypoints_flat.unsqueeze(1) - keypoints_flat.unsqueeze(0)  # Shape: (N, N, K*2)
    distances = torch.norm(diff, dim=2)  # Shape: (N, N)

    # Create a mask for distances less than the threshold
    distance_mask = distances < keypoint_threshold
    distance_mask.fill_diagonal_(False)  # Exclude self-comparisons

    # Create a mask to consider only detections with lower scores
    idx_i = torch.arange(N).unsqueeze(1).expand(N, N).to(device)
    idx_j = torch.arange(N).unsqueeze(0).expand(N, N).to(device)
    upper_triangle = idx_j > idx_i

    # Final suppression mask
    suppression_mask = distance_mask & upper_triangle

    # Determine which detections to suppress
    suppressed = suppression_mask.any(dim=0)

    # Indices of detections to keep after keypoint suppression
    keypoint_keep_indices = torch.nonzero(~suppressed, as_tuple=False).view(-1)

    # Update detections after keypoint suppression
    nms_bboxes = nms_bboxes[keypoint_keep_indices]
    nms_scores = sorted_scores[keypoint_keep_indices]
    nms_labels = nms_labels[keypoint_keep_indices]
    nms_keypoints = nms_keypoints[keypoint_keep_indices]

    # Step 3: Suppress Inner Bounding Boxes
    N_final = nms_bboxes.shape[0]

    # If no detections remain, return empty results
    if N_final == 0:
        return results[keep_indices]

    x1 = nms_bboxes[:, 0].unsqueeze(1)  # Shape: (N_final, 1)
    y1 = nms_bboxes[:, 1].unsqueeze(1)
    x2 = nms_bboxes[:, 2].unsqueeze(1)
    y2 = nms_bboxes[:, 3].unsqueeze(1)

    x1_j = nms_bboxes[:, 0].unsqueeze(0)  # Shape: (1, N_final)
    y1_j = nms_bboxes[:, 1].unsqueeze(0)
    x2_j = nms_bboxes[:, 2].unsqueeze(0)
    y2_j = nms_bboxes[:, 3].unsqueeze(0)

    # Check if box_i is entirely inside box_j
    x1_in = x1 >= x1_j  # Shape: (N_final, N_final)
    y1_in = y1 >= y1_j
    x2_in = x2 <= x2_j
    y2_in = y2 <= y2_j

    # Combine conditions
    inside_mask = x1_in & y1_in & x2_in & y2_in
    inside_mask.fill_diagonal_(False)  # Exclude self-comparisons

    # Create a mask to consider only detections with lower scores
    idx_i = torch.arange(N_final).unsqueeze(1).expand(N_final, N_final).to(device)
    idx_j = torch.arange(N_final).unsqueeze(0).expand(N_final, N_final).to(device)
    upper_triangle = idx_j > idx_i

    # Final suppression mask for inner boxes
    inner_box_suppression_mask = inside_mask & upper_triangle

    # Determine which detections to suppress based on inner boxes
    inner_box_suppressed = inner_box_suppression_mask.any(dim=0)

    # Indices of detections to keep after inner box suppression
    inner_box_keep_indices = torch.nonzero(~inner_box_suppressed, as_tuple=False).view(-1)

    # Update detections after inner box suppression
    final_bboxes = nms_bboxes[inner_box_keep_indices]
    final_scores = nms_scores[inner_box_keep_indices]
    final_labels = nms_labels[inner_box_keep_indices]
    final_keypoints = nms_keypoints[inner_box_keep_indices]

    # Prepare final results
    nms_results = InstanceData()
    nms_results.bboxes = final_bboxes
    nms_results.scores = final_scores
    nms_results.labels = final_labels
    nms_results.keypoints = final_keypoints

    return nms_results


def keypoint_nms1(results: InstanceData) -> InstanceData:
    
    bboxes = results.bboxes
    scores = results.scores
    keypoints = results.keypoints
    labels = results.labels

    device = bboxes.device

    iou_threshold = 0.3  # Adjust as needed
    keypoint_threshold = 5.0  # Adjust based on the scale of your keypoint coordinates

    keep_indices = nms(bboxes, scores, iou_threshold)

    nms_bboxes = bboxes[keep_indices]
    nms_scores = scores[keep_indices]
    nms_labels = labels[keep_indices]
    nms_keypoints = keypoints[keep_indices]

    N = nms_keypoints.shape[0]
    keypoints_flat = nms_keypoints.view(N, -1)  # Shape: (N, 6) for 3 keypoints with (x, y)

    sorted_scores, sorted_indices = torch.sort(nms_scores, descending=True)
    keypoints_flat = keypoints_flat[sorted_indices]

    diff = keypoints_flat.unsqueeze(1) - keypoints_flat.unsqueeze(0)  # Shape: (N, N, 6)
    distances = torch.norm(diff, dim=2)  # Shape: (N, N)

    # Create a mask for distances less than the threshold
    distance_mask = distances < keypoint_threshold
    # Zero out diagonal (distance with self)
    distance_mask.fill_diagonal_(False)

    # Create a mask to consider only detections with lower scores (since we sorted scores descending)
    idx_i = torch.arange(N).unsqueeze(1).expand(N, N).to(bboxes.device)
    idx_j = torch.arange(N).unsqueeze(0).expand(N, N).to(bboxes.device)
    # Create mask where j > i
    upper_triangle = idx_j > idx_i

    # Final suppression mask
    suppression_mask = distance_mask & upper_triangle

    # Determine which detections to suppress
    suppressed = suppression_mask.any(dim=0)

    # Indices of detections to keep after keypoint suppression
    keypoint_keep_indices = torch.nonzero(~suppressed, as_tuple=False).view(-1)

    # Map back to original indices
    final_keep_indices = keep_indices[sorted_indices[keypoint_keep_indices]]

    kpt_sup_bboxes = results[final_keep_indices].bboxes

    N_final = kpt_sup_bboxes.shape[0]
    x1 = kpt_sup_bboxes[:, 0].unsqueeze(1)  # Shape: (N_final, 1)
    y1 = kpt_sup_bboxes[:, 1].unsqueeze(1)
    x2 = kpt_sup_bboxes[:, 2].unsqueeze(1)
    y2 = kpt_sup_bboxes[:, 3].unsqueeze(1)

    x1_j = kpt_sup_bboxes[:, 0].unsqueeze(0)  # Shape: (1, N_final)
    y1_j = kpt_sup_bboxes[:, 1].unsqueeze(0)
    x2_j = kpt_sup_bboxes[:, 2].unsqueeze(0)
    y2_j = kpt_sup_bboxes[:, 3].unsqueeze(0)

    # Check if box_i is entirely inside box_j
    x1_in = x1 >= x1_j  # Shape: (N_final, N_final)
    y1_in = y1 >= y1_j
    x2_in = x2 <= x2_j
    y2_in = y2 <= y2_j

    # Combine conditions
    inside_mask = x1_in & y1_in & x2_in & y2_in
    inside_mask.fill_diagonal_(False)  # Exclude self-comparisons

    # Create a mask to consider only detections with lower scores
    # Since the detections are already sorted by descending score, we can reuse indices
    idx_i = torch.arange(N_final).unsqueeze(1).expand(N_final, N_final).to(device)
    idx_j = torch.arange(N_final).unsqueeze(0).expand(N_final, N_final).to(device)
    upper_triangle = idx_j > idx_i

    # Final suppression mask for inner boxes
    inner_box_suppression_mask = inside_mask & upper_triangle

    # Determine which detections to suppress based on inner boxes
    inner_box_suppressed = inner_box_suppression_mask.any(dim=0)

    # Indices of detections to keep after inner box suppression
    inner_box_keep_indices = torch.nonzero(~inner_box_suppressed, as_tuple=False).view(-1)

    final_keep_indices = final_keep_indices[inner_box_keep_indices]

    nms_results = results[final_keep_indices]

    return nms_results


    print(scores.shape, labels.shape, keypoints.shape, bboxes.shape)
    
    # exit()
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # print(nms_cfg)
    # print(boxes.shape, scores.shape, idxs.shape)

    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop('type', 'nms')
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep
