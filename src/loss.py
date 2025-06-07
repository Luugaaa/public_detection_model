import torch
import torch.nn as nn
from torchvision.ops import box_area

def giou_loss(pred_boxes, target_boxes):
    """Generalized IoU loss for bounding boxes (xyxy format)"""
    # Ensure boxes are properly formed (x2 > x1, y2 > y1)
    pred_boxes_fixed = torch.zeros_like(pred_boxes)
    pred_boxes_fixed[:, 0] = torch.min(pred_boxes[:, 0], pred_boxes[:, 2])
    pred_boxes_fixed[:, 1] = torch.min(pred_boxes[:, 1], pred_boxes[:, 3])
    pred_boxes_fixed[:, 2] = torch.max(pred_boxes[:, 0], pred_boxes[:, 2])
    pred_boxes_fixed[:, 3] = torch.max(pred_boxes[:, 1], pred_boxes[:, 3])
    
    # Calculate IoU
    inter_x1 = torch.max(pred_boxes_fixed[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes_fixed[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes_fixed[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes_fixed[:, 3], target_boxes[:, 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    pred_area = (pred_boxes_fixed[:, 2] - pred_boxes_fixed[:, 0]) * (pred_boxes_fixed[:, 3] - pred_boxes_fixed[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    # Enclosing box
    enclose_x1 = torch.min(pred_boxes_fixed[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes_fixed[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes_fixed[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes_fixed[:, 3], target_boxes[:, 3])
    
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose_area = enclose_w * enclose_h
    
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    giou = giou.clamp(min=-1.0, max=1.0)
    return (1 - giou).clamp(min=0).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.eps = 1e-7
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # Standard cross entropy loss
        ce_loss = self.ce(inputs, targets)
        # Get predicted probabilities for the target class
        # This is more accurate than using exp(-ce_loss)
        with torch.no_grad():
            probs = torch.softmax(inputs, dim=1)
            idx = torch.arange(probs.size(0))
            target_probs = probs[idx.numpy(), targets.cpu().numpy()]
            pt = target_probs + self.eps  
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha if needed
        if self.alpha is not None:
            if self.num_classes:
                # For multi-class, create a tensor of alpha weights
                alpha = torch.ones(self.num_classes, device=inputs.device) * (1 - self.alpha)
                
                alpha[0] = self.alpha  # assuming class 0 is the background class # TODO : change that
                alpha_t = alpha[targets.cpu().numpy()]
            else:
                # For binary classification
                alpha_t = self.alpha * (targets == 1).float() + (1 - self.alpha) * (targets == 0).float()
                
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class DetectionLoss(nn.Module):
    def __init__(self, num_classes, image_size=224, obj_weight=0.1, box_weight=1.0, cls_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_loss = FocalLoss(num_classes=num_classes)
        self.bce_loss = nn.BCELoss(reduction="none")
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        
    def forward(self, preds, targets):
        box_loss = 0
        cls_loss = 0
        obj_loss = 0
        valid_targets = 0
        
        # Handle batch of predictions
        for batch_idx, (pred, target) in enumerate(zip(preds, targets)):
            # Handle objectness loss for all predictions regardless of target content
            obj_pred = preds['objectness'][batch_idx]
            batch_obj_loss = self._objectness_loss(obj_pred, target)
            obj_loss += batch_obj_loss * self.obj_weight
            
            # Skip box and class loss calculation if no objects in target
            if len(target['boxes']) == 0:
                valid_targets += 1  # Still count as valid for objectness
                continue
                
            # Box loss: compute IoU/GIoU between predictions and ground truth
            # print("IN : ", pred, "\n TAR : ", target)
            box_pred = preds['boxes'][batch_idx]
            batch_box_loss = self._box_loss(box_pred, obj_pred, target)
            box_loss += batch_box_loss * self.box_weight
            
            # Classification loss
            cls_pred = preds['classes'][batch_idx]
            batch_cls_loss = self._class_loss(cls_pred, obj_pred, target)
            cls_loss += batch_cls_loss * self.cls_weight
            
            valid_targets += 1
            
        if valid_targets == 0:
            return (torch.tensor(0., device=pred['boxes'].device), 
                    torch.tensor(0., device=pred['boxes'].device),
                    torch.tensor(0., device=pred['boxes'].device))
        
        # Return the weighted average losses
        return (box_loss / valid_targets, 
                cls_loss / valid_targets,
                obj_loss / valid_targets)
    
    def _objectness_loss(self, obj_pred, target):
        """Calculate objectness loss for both positive and negative examples"""
        # Get objectness predictions
          # [B, max_objects, H, W]
        max_objects, H, W = obj_pred.shape
        
        # Create objectness target tensor (initially all zeros)
        obj_target = torch.zeros_like(obj_pred)
        
        if len(target['boxes']) > 0:
            # Create spatial grid
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=obj_pred.device),
                                           torch.arange(W, device=obj_pred.device),
                                           indexing='ij')
            
            # Convert from YOLO format to grid coordinates
            # YOLO format is [center_x, center_y, width, height]
            gt_cx, gt_cy = target['boxes'][:, 0], target['boxes'][:, 1]
            
            # Convert to grid coordinates
            gt_cx_grid = (gt_cx * W).long().clamp(0, W-1)
            gt_cy_grid = (gt_cy * H).long().clamp(0, H-1)
            
            # For each ground truth box, set the corresponding grid cell's objectness to 1
            # Use the first available object slot at each position
            for i in range(len(target['boxes'])):
                cx, cy = gt_cx_grid[i], gt_cy_grid[i]
                
                # Find the first available object slot at this position
                for obj_idx in range(max_objects):
                    if obj_target[obj_idx, cy, cx] == 0:
                        obj_target[obj_idx, cy, cx] = 1
                        break
        
        # Calculate binary cross entropy loss for objectness
        obj_loss = self.bce_loss(obj_pred, obj_target)
        
        # Apply higher weight for positive examples (focal-like weighting)
        pos_weight = 10.0  # Weight for positive examples
        obj_weight = torch.ones_like(obj_target)
        obj_weight[obj_target > 0] = pos_weight
        
        obj_loss = obj_loss * obj_weight
        return obj_loss.mean()

    def _box_loss(self, box_pred, obj_pred, target):
        """Calculate box regression loss for positive examples only"""
        # Get box predictions and reshape
          # [B, max_objects, 4, H, W]
        max_objects, _, H, W = box_pred.shape
        
        # Create a list to store matched predictions
        matched_box_preds = []
        matched_box_targets = []
        
        if len(target['boxes']) > 0:
            # Get ground truth boxes (in YOLO format)
            gt_boxes = target['boxes']  # [num_targets, 4] - [cx, cy, w, h]
            
            # Calculate grid positions of ground truth centers
            gt_cx, gt_cy = gt_boxes[:, 0], gt_boxes[:, 1]
            gt_cx_grid = (gt_cx * W).long().clamp(0, W-1)
            gt_cy_grid = (gt_cy * H).long().clamp(0, H-1)
            
            # For each ground truth, get the corresponding prediction
            for i in range(len(gt_boxes)):
                cx, cy = gt_cx_grid[i], gt_cy_grid[i]
                
                # Get predictions at this location (all object slots)
                loc_preds = box_pred[:, :, cy, cx]  # [max_objects, 4]
                
                # Find best matching prediction slot
                # For now, just use the first slot where objectness > 0.5 or the first slot TODO : update this
                obj_scores = obj_pred[:, cy, cx]  # [max_objects]
                positive_slots = (obj_scores > 0.5).nonzero(as_tuple=True)[0]
                
                if len(positive_slots) > 0:
                    obj_idx = positive_slots[0]
                else:
                    obj_idx = 0
                
                matched_pred = loc_preds[obj_idx]  # [4]
                matched_box_preds.append(matched_pred)
                matched_box_targets.append(gt_boxes[i])
        
        if not matched_box_preds:
            return torch.tensor(0., device=box_pred.device)
        
        # Stack matched predictions and targets
        matched_box_preds = torch.stack(matched_box_preds)  # [num_targets, 4]
        matched_box_targets = torch.stack(matched_box_targets)  # [num_targets, 4]
        
        # Convert YOLO format to xyxy format for IoU calculation
        def yolo_to_xyxy(boxes):
            # boxes: [N, 4] in format [cx, cy, w, h]
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return torch.stack([x1, y1, x2, y2], dim=1)
        
        pred_xyxy = yolo_to_xyxy(matched_box_preds)
        target_xyxy = yolo_to_xyxy(matched_box_targets)
        
        # Calculate GIoU loss
        return giou_loss(pred_xyxy, target_xyxy)

    def _class_loss(self, cls_pred, obj_pred, target):
        """Calculate classification loss for positive examples only"""
        # Get class predictions
          # [B, max_objects, num_classes, H, W]
        # max_objects, num_classes, H, W = cls_pred.shape
        
        # Create lists to store matched predictions and targets
        matched_cls_preds = []
        matched_cls_targets = []
        
        if len(target['boxes']) > 0:
            # Get ground truth boxes and labels
            gt_boxes = target['boxes']  # [num_targets, 4]
            gt_labels = target['labels']  # [num_targets]
            
            # Calculate grid positions of ground truth centers
            gt_cx, gt_cy = gt_boxes[:, 0], gt_boxes[:, 1]
            gt_cx_grid = (gt_cx * W).long().clamp(0, W-1)
            gt_cy_grid = (gt_cy * H).long().clamp(0, H-1)
            
            # For each ground truth, get the corresponding prediction
            for i in range(len(gt_boxes)):
                cx, cy = gt_cx_grid[i], gt_cy_grid[i]
                label = gt_labels[i]
                
                # Get predictions at this location (all object slots)
                loc_preds = cls_pred[:, :, cy, cx]  # [max_objects, num_classes]
                
                obj_scores = obj_pred[:, cy, cx]  # [max_objects]
                positive_slots = (obj_scores > 0.5).nonzero(as_tuple=True)[0]
                
                if len(positive_slots) > 0:
                    obj_idx = positive_slots[0]
                else:
                    obj_idx = 0
                
                matched_pred = loc_preds[obj_idx]  # [num_classes]
                matched_cls_preds.append(matched_pred)
                matched_cls_targets.append(label)
        
        if not matched_cls_preds:
            return torch.tensor(0., device=cls_pred.device)
        
        # Stack matched predictions and targets
        matched_cls_preds = torch.stack(matched_cls_preds)  # [num_targets, num_classes]
        matched_cls_targets = torch.stack(matched_cls_targets).to(device=cls_pred.device, dtype=torch.long)
        # Calculate focal loss
        return self.focal_loss(matched_cls_preds, matched_cls_targets)
