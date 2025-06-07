import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- IoU Utilities (box_cxcywh_to_xyxy, box_iou, ciou_loss) ---
def box_cxcywh_to_xyxy(box_cxcywh): 
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = cx - 0.5 * w; y1 = cy - 0.5 * h; x2 = cx + 0.5 * w; y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)

def box_iou(boxes1_xyxy, boxes2_xyxy):
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2]); rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0); intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - intersection + 1e-6
    return intersection / union, union

def ciou_loss(pred_boxes_xyxy, target_boxes_xyxy, eps=1e-7):
    iou, _ = box_iou(pred_boxes_xyxy, target_boxes_xyxy); iou = iou.diag()
    pred_cxcy = (pred_boxes_xyxy[:, :2] + pred_boxes_xyxy[:, 2:]) / 2; target_cxcy = (target_boxes_xyxy[:, :2] + target_boxes_xyxy[:, 2:]) / 2
    center_dist_sq = (pred_cxcy - target_cxcy).pow(2).sum(dim=-1)
    enclose_mins = torch.min(pred_boxes_xyxy[:, :2], target_boxes_xyxy[:, :2]); enclose_maxs = torch.max(pred_boxes_xyxy[:, 2:], target_boxes_xyxy[:, 2:])
    enclose_wh = (enclose_maxs - enclose_mins).clamp(min=0); enclose_diag_sq = enclose_wh.pow(2).sum(dim=-1) + eps
    diou_penalty = center_dist_sq / enclose_diag_sq
    pred_w_h = pred_boxes_xyxy[:, 2:] - pred_boxes_xyxy[:, :2]; target_w_h = target_boxes_xyxy[:, 2:] - target_boxes_xyxy[:, :2]
    v = (4 / (math.pi**2)) * (torch.atan(target_w_h[:, 0] / (target_w_h[:, 1] + eps)) - torch.atan(pred_w_h[:, 0] / (pred_w_h[:, 1] + eps))).pow(2)
    with torch.no_grad(): alpha = v / (1 - iou + v + eps)
    return 1 - (iou - diou_penalty - alpha * v)


# --- Focal Loss ---
class FocalLoss(nn.Module): 
    def __init__(self, loss_fcn, gamma=2.0, alpha=0.25):
        super().__init__(); assert loss_fcn.reduction == 'none'; self.loss_fcn = loss_fcn
        self.gamma = gamma; self.alpha = alpha; self.epsilon = 1e-6
    def forward(self, pred_logits, target):
        bce_loss = self.loss_fcn(pred_logits, target.float()); p = torch.sigmoid(pred_logits)
        p_t = p * target + (1 - p) * (1 - target); p_t = p_t.clamp(min=self.epsilon, max=1.0 - self.epsilon)
        modulating_factor = (1.0 - p_t)**self.gamma; alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * modulating_factor * bce_loss).sum()


class DFL_CIoU_Loss(nn.Module):
    def __init__(self, num_classes, reg_max=16,
                 lambda_box_iou=7.5, lambda_box_dfl=1.5,
                 lambda_cls=0.5, lambda_obj=3.0,
                 focal_loss_gamma=2.0, focal_loss_alpha=0.25,
                 image_size_for_ciou_norm = 640):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.lambda_box_iou = lambda_box_iou
        self.lambda_box_dfl = lambda_box_dfl
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.image_size_for_ciou_norm = image_size_for_ciou_norm

        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.obj_criterion = FocalLoss(nn.BCEWithLogitsLoss(reduction='none'), gamma=focal_loss_gamma, alpha=focal_loss_alpha)
        self.dfl_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.project = torch.arange(reg_max, dtype=torch.float32)

    def forward(self, predictions, targets):
        pred_dfl_logits = predictions['boxes_dfl']
        pred_cls_logits = predictions['classes']
        pred_obj_logits = predictions['objectness']

        target_boxes_cxcywh_norm = targets['target_boxes']
        target_ltrb_stride_units = targets['target_ltrb_for_dfl']
        target_classes = targets['target_classes']
        target_objectness = targets['target_objectness']
        pos_mask = targets['pos_mask']
        anchor_points_norm = targets['anchor_points_norm']
        strides_tensor = targets['strides_tensor']

        B, N_total, _ = pred_cls_logits.shape
        device = pred_cls_logits.device
        if self.project.device != device: self.project = self.project.to(device)

        if target_objectness.dim() == 2:
            target_objectness = target_objectness.unsqueeze(-1)

        actual_num_pos = pos_mask.sum() # True number of positive samples
        num_pos_for_norm = actual_num_pos.clamp(min=1.0) # For loss normalization denominator

        loss_obj_sum = self.obj_criterion(pred_obj_logits.view(-1), target_objectness.view(-1))

        pred_cls_pos = pred_cls_logits[pos_mask]
        target_cls_pos = target_classes[pos_mask]
        pred_dfl_logits_pos = pred_dfl_logits[pos_mask] # Shape (actual_num_pos, 4 * reg_max)
        target_ltrb_pos = target_ltrb_stride_units[pos_mask]
        anchor_points_pos = anchor_points_norm[pos_mask]
        strides_pos = strides_tensor[pos_mask]
        target_boxes_cxcywh_pos = target_boxes_cxcywh_norm[pos_mask]

        loss_cls_sum = torch.tensor(0.0, device=device)
        loss_dfl_sum = torch.tensor(0.0, device=device)
        loss_iou_sum = torch.tensor(0.0, device=device)

        if actual_num_pos > 0: # Guard operations on positive samples
            # print(f"Loss function: actual_num_pos = {actual_num_pos.item()}")
            _actual_num_pos_int = actual_num_pos.item() # Python int for view()

            loss_cls_sum = self.cls_criterion(pred_cls_pos, target_cls_pos.float()).sum()
            
            # DFL logits for positive predictions, reshaped for DFL loss calculation
            # Original shape of pred_dfl_logits_pos is (actual_num_pos, 4 * reg_max)
            # Reshape to (actual_num_pos * 4, reg_max) for DFL processing
            pred_ltrb_dist_for_dfl_loss = pred_dfl_logits_pos.view(-1, self.reg_max)
            
            tl = target_ltrb_pos.view(-1) # (actual_num_pos * 4)
            tl_int_low = tl.floor().long()
            tl_int_high = tl.ceil().long()
            wl = (tl_int_high.float() - tl).to(pred_ltrb_dist_for_dfl_loss.dtype)
            wh = (tl - tl_int_low.float()).to(pred_ltrb_dist_for_dfl_loss.dtype)

            idx1 = tl_int_low.unsqueeze(1).clamp(0, self.reg_max - 1) # (actual_num_pos*4, 1)
            idx2 = tl_int_high.unsqueeze(1).clamp(0, self.reg_max - 1) # (actual_num_pos*4, 1)
            
            src1 = wl.unsqueeze(1) # (actual_num_pos*4, 1)
            src2 = wh.unsqueeze(1) # (actual_num_pos*4, 1)

            target_label_dist = torch.zeros_like(pred_ltrb_dist_for_dfl_loss, device=device) # (actual_num_pos*4, reg_max)
            
            is_integer_target = (tl == tl_int_low.float()) # (actual_num_pos*4,)
            non_integer_mask = ~is_integer_target

            if non_integer_mask.any():
                current_ni_idx1 = idx1[non_integer_mask]
                current_ni_src1 = src1[non_integer_mask]
                current_ni_idx2 = idx2[non_integer_mask]
                current_ni_src2 = src2[non_integer_mask]
                ni_target_slice = target_label_dist[non_integer_mask]
                ni_target_slice.scatter_(1, current_ni_idx1, current_ni_src1)
                ni_target_slice.scatter_(1, current_ni_idx2, current_ni_src2)
                target_label_dist[non_integer_mask] = ni_target_slice

            if is_integer_target.any():
                current_int_idx = idx1[is_integer_target]
                int_target_slice = target_label_dist[is_integer_target]
                one_val_src = torch.ones_like(current_int_idx, dtype=target_label_dist.dtype)
                int_target_slice.zero_()
                int_target_slice.scatter_(1, current_int_idx, one_val_src)
                target_label_dist[is_integer_target] = int_target_slice
            
            bce_loss_for_dfl = self.dfl_criterion(pred_ltrb_dist_for_dfl_loss, target_label_dist)
            loss_dfl_sum = (bce_loss_for_dfl * target_label_dist).sum()

            # --- CIoU Loss Calculation ---
            # Decode predicted boxes from DFL distribution for CIoU
            # Use pred_dfl_logits_pos (actual_num_pos, 4 * reg_max)
            # Reshape to (actual_num_pos, 4, reg_max)
            reshaped_for_softmax = pred_dfl_logits_pos.view(_actual_num_pos_int, 4, self.reg_max) # Corrected view
            pred_ltrb_distributions_softmaxed = F.softmax(reshaped_for_softmax, dim=2) # (actual_num_pos, 4, reg_max)
            
            pred_ltrb_stride_units = pred_ltrb_distributions_softmaxed.matmul(self.project) # (actual_num_pos, 4)

            pixel_offsets = pred_ltrb_stride_units * strides_pos # (actual_num_pos, 4) in pixel units
            norm_offsets = pixel_offsets / self.image_size_for_ciou_norm # (actual_num_pos, 4) in normalized img units

            pred_x1_norm = anchor_points_pos[:, 0] - norm_offsets[:, 0]
            pred_y1_norm = anchor_points_pos[:, 1] - norm_offsets[:, 1]
            pred_x2_norm = anchor_points_pos[:, 0] + norm_offsets[:, 2]
            pred_y2_norm = anchor_points_pos[:, 1] + norm_offsets[:, 3]
            pred_boxes_xyxy_norm = torch.stack([pred_x1_norm, pred_y1_norm, pred_x2_norm, pred_y2_norm], dim=-1)

            target_boxes_xyxy_pos = box_cxcywh_to_xyxy(target_boxes_cxcywh_pos)
            loss_iou_values = ciou_loss(pred_boxes_xyxy_norm, target_boxes_xyxy_pos)
            loss_iou_sum = loss_iou_values.sum()

        box_loss_dfl = (self.lambda_box_dfl * loss_dfl_sum) / num_pos_for_norm
        box_loss_iou = (self.lambda_box_iou * loss_iou_sum) / num_pos_for_norm
        cls_loss = (self.lambda_cls * loss_cls_sum) / num_pos_for_norm
        obj_loss = (self.lambda_obj * loss_obj_sum) / num_pos_for_norm

        total_loss = box_loss_dfl + box_loss_iou + cls_loss + obj_loss

        return {
            'total_loss': total_loss,
            'box_iou_loss': box_loss_iou.detach(),
            'box_dfl_loss': box_loss_dfl.detach(),
            'cls_loss': cls_loss.detach(),
            'obj_loss': obj_loss.detach(),
            'num_pos': actual_num_pos # Log the true number of positives
        }