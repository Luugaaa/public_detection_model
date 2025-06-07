import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.ops as ops
import torchmetrics
from tqdm import tqdm
from typing import Type
import albumentations as A
import cv2
import traceback 
import os
import torch.nn.functional as F


from src.detection_model_architecture import DetectionModel
from src.dataset_handlers.full_coco_dataset_handler import CocoDataset
from src.save_tools import load_model
from src.config.config import CLASS_NAMES
from src.collate_fn import collate_fn


def generate_anchor_points_and_strides(strides, img_h, img_w, k=1, device='cpu'):
    """
    Generates anchor points (normalized centers) and strides for each prediction slot.
    """
    all_anchor_points = []
    all_strides = []
    total_preds_calculated = 0
    for stride in strides:
        h_grid = img_h // stride
        w_grid = img_w // stride
        if h_grid <= 0 or w_grid <= 0: continue

        num_cells = h_grid * w_grid
        num_preds_scale = num_cells * k

        gy, gx = torch.meshgrid(torch.arange(h_grid, device=device), torch.arange(w_grid, device=device), indexing='ij')
        # Normalized anchor points (cell centers) for this scale
        anchors_scale = torch.stack([(gx.flatten() + 0.5) / w_grid, (gy.flatten() + 0.5) / h_grid], dim=-1).float()
        strides_scale = torch.full((num_cells, 1), stride, device=device, dtype=torch.float32)

        all_anchor_points.append(anchors_scale.repeat_interleave(k, dim=0))
        all_strides.append(strides_scale.repeat_interleave(k, dim=0))
        total_preds_calculated += num_preds_scale

    if not all_anchor_points:
        # print("Warning: No valid anchors generated in generate_anchor_points_and_strides")
        return torch.empty((0, 2), device=device), torch.empty((0, 1), device=device)

    anchor_points_norm = torch.cat(all_anchor_points, dim=0)
    strides_map = torch.cat(all_strides, dim=0)
    # print(f"Generated {anchor_points_norm.shape[0]} anchor points and strides.")
    return anchor_points_norm, strides_map

def decode_dfl_predictions(outputs, anchor_points_norm, strides_map, reg_max, img_h, img_w):
    """
    Decodes raw DFL model outputs into bounding boxes, scores, and labels.
    Outputs boxes in absolute pixel coordinates [x1, y1, x2, y2].
    """
    pred_dfl_logits = outputs['boxes_dfl']   # (B, N_total, 4 * reg_max)
    pred_cls_logits = outputs['classes']       # (B, N_total, C)
    pred_obj_logits = outputs['objectness']    # (B, N_total, 1)
    device = pred_dfl_logits.device
    B, N_total_preds, _ = pred_cls_logits.shape # Use N from class predictions

    # Align N_total from anchors/strides with N_total from predictions
    N_total_anchors = anchor_points_norm.shape[0]
    
    if N_total_anchors != N_total_preds:
        # print(f"Decode Warning: Anchor count ({N_total_anchors}) != Prediction count ({N_total_preds}). Using {N_total_preds}.")
        if N_total_anchors > N_total_preds:
             anchor_points_norm = anchor_points_norm[:N_total_preds]
             strides_map = strides_map[:N_total_preds]
             N_total = N_total_preds
        else: # Predictions > Anchors? Problem!
             print(f"Decode ERROR: More predictions ({N_total_preds}) than generated anchors ({N_total_anchors})!")
             # Fallback: use anchor count, slice predictions (data loss!)
             N_total = N_total_anchors # Adjust N_total
             pred_dfl_logits = pred_dfl_logits[:, :N_total_anchors, :]
             pred_cls_logits = pred_cls_logits[:, :N_total_anchors, :]
             pred_obj_logits = pred_obj_logits[:, :N_total_anchors, :]
    else : 
        N_total = N_total_preds

    # Expand anchors/strides for batch dimension
    anchor_points_norm = anchor_points_norm.unsqueeze(0).expand(B, N_total, -1) # (B, N_total, 2)
    strides_map = strides_map.unsqueeze(0).expand(B, N_total, -1)          # (B, N_total, 1)

    # Decode DFL to ltrb (stride units)
    project_tensor = torch.arange(reg_max, device=device, dtype=torch.float32)
    reshaped_dfl = pred_dfl_logits.view(B, N_total, 4, reg_max)
    distributions = F.softmax(reshaped_dfl, dim=3)
    pred_ltrb_stride_units = distributions.matmul(project_tensor) # (B, N_total, 4)

    # Convert ltrb (stride units) to xyxy (absolute pixels)
    s = strides_map # (B, N_total, 1)
    anchor_cx_norm = anchor_points_norm[..., 0:1] # (B, N_total, 1)
    anchor_cy_norm = anchor_points_norm[..., 1:2] # (B, N_total, 1)

    l_stride, t_stride, r_stride, b_stride = pred_ltrb_stride_units.chunk(4, dim=-1)

    # Offsets in pixels
    l_offset_px = l_stride * s; t_offset_px = t_stride * s
    r_offset_px = r_stride * s; b_offset_px = b_stride * s

    # Anchor points in pixels
    anchor_cx_px = anchor_cx_norm * img_w; anchor_cy_px = anchor_cy_norm * img_h

    # Absolute coordinates
    pred_x1 = anchor_cx_px - l_offset_px; pred_y1 = anchor_cy_px - t_offset_px
    pred_x2 = anchor_cx_px + r_offset_px; pred_y2 = anchor_cy_px + b_offset_px

    decoded_boxes_abs_xyxy = torch.cat([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
    # Clamp to image boundaries AFTER calculating coordinates
    decoded_boxes_abs_xyxy[:, :, 0::2] = decoded_boxes_abs_xyxy[:, :, 0::2].clamp(0, img_w)
    decoded_boxes_abs_xyxy[:, :, 1::2] = decoded_boxes_abs_xyxy[:, :, 1::2].clamp(0, img_h)

    # Calculate scores
    obj_prob = torch.sigmoid(pred_obj_logits).squeeze(-1)
    cls_prob = torch.sigmoid(pred_cls_logits) # Assuming BCE base for classes
    scores, labels = torch.max(cls_prob, dim=-1)
    final_scores = scores * obj_prob

    return {
        "boxes": decoded_boxes_abs_xyxy, # Absolute pixel coords [x1, y1, x2, y2]
        "scores": final_scores,
        "labels": labels
    }

def post_process_batch(decoded_preds, conf_thres=0.25, iou_thres=0.45, min_box_area=1.0):
    """ Applies confidence thresholding and NMS. Boxes are absolute pixels. """
    batch_boxes_abs = decoded_preds['boxes']    # (B, N, 4)
    batch_scores = decoded_preds['scores']      # (B, N)
    batch_labels = decoded_preds['labels']      # (B, N)
    B = batch_boxes_abs.shape[0]
    device = batch_boxes_abs.device
    output_list = []

    for i in range(B):
        boxes = batch_boxes_abs[i]; scores = batch_scores[i]; labels = batch_labels[i]
        keep_conf = scores >= conf_thres
        boxes, scores, labels = boxes[keep_conf], scores[keep_conf], labels[keep_conf]

        if boxes.shape[0] == 0:
            output_list.append({'boxes': boxes, 'scores': scores, 'labels': labels}); continue

        # NMS expects FloatTensor
        keep_nms = ops.batched_nms(boxes.float(), scores, labels, iou_thres)
        boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

        if min_box_area > 0.0:
            box_widths = boxes[:, 2] - boxes[:, 0]
            box_heights = boxes[:, 3] - boxes[:, 1]
            areas = box_widths * box_heights
            keep_area = areas >= min_box_area
            boxes, scores, labels = boxes[keep_area], scores[keep_area], labels[keep_area]
        
        max_detections_limit = 300 # Or 100, align with YOLOv8
        if boxes.shape[0] > max_detections_limit:
            indices = torch.argsort(scores, descending=True)[:max_detections_limit]
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

        output_list.append({'boxes': boxes, 'scores': scores, 'labels': labels})
    return output_list

def format_targets_for_metric(raw_targets_list, img_h, img_w, device='cpu'):
    """ Formats GT targets for torchmetrics: absolute xyxy boxes. """
    formatted_targets = []
    if raw_targets_list is None: return formatted_targets

    for i, target_data in enumerate(raw_targets_list):
        if not isinstance(target_data, dict):
            print(f"\nERROR formatting target {i}: Not a dict. Value: {target_data}")
            formatted_targets.append({'boxes': torch.empty((0, 4), dtype=torch.float32, device=device),
                                      'labels': torch.empty((0,), dtype=torch.long, device=device)}); continue

        gt_boxes_yolo = target_data.get('boxes'); gt_labels = target_data.get('labels')

        if gt_boxes_yolo is None or gt_labels is None or not isinstance(gt_boxes_yolo, torch.Tensor) or not isinstance(gt_labels, torch.Tensor):
             # print(f"Warning: Target {i} missing 'boxes'/'labels' tensors. Adding empty.")
             formatted_targets.append({'boxes': torch.empty((0, 4), dtype=torch.float32, device=device),
                                      'labels': torch.empty((0,), dtype=torch.long, device=device)}); continue

        target_device = gt_boxes_yolo.device # Use device of source tensor
        if gt_boxes_yolo.shape[0] == 0:
            formatted_targets.append({'boxes': torch.empty((0, 4), dtype=torch.float32, device=target_device),
                                      'labels': torch.empty((0,), dtype=torch.long, device=target_device)}); continue

        gt_boxes_yolo = gt_boxes_yolo.float(); gt_labels = gt_labels.long()
        cx, cy, w, h = gt_boxes_yolo.unbind(-1)
        x1 = (cx - w / 2) * img_w; y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w; y2 = (cy + h / 2) * img_h
        abs_boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0) # Clamp min to 0
        # Further clamp max to img bounds
        abs_boxes[:, 0::2] = abs_boxes[:, 0::2].clamp(max=img_w)
        abs_boxes[:, 1::2] = abs_boxes[:, 1::2].clamp(max=img_h)

        formatted_targets.append({'boxes': abs_boxes.to(device), 'labels': gt_labels.to(device)}) # Ensure output on target device
    return formatted_targets

# ==============================================================================
# Main Evaluator Function (Updated for DFL Model)
# ==============================================================================
def evaluate_model(
    model_path: str,
    model_type: Type[nn.Module], 
    data_loader: DataLoader, 
    dataset_name: str = "Validation", 
    # --- Model & Data Config ---
    class_names: list = CLASS_NAMES, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    img_size: int = 640, 
    strides: list = [8, 16, 32],
    max_objects_per_pixel: int = 1, # K value
    # --- Evaluation Config ---
    conf_thres: float = 0.001, # Low confidence for mAP calculation
    iou_thres: float = 0.60,   # NMS IoU threshold (COCO standard often uses 0.6 or 0.65)
    min_box_area: float = 1.0,
):
    eval_device = torch.device(device)

    # --- 1. Load Model ---
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path, modeltype=model_type, device=eval_device)
        model.eval()
        num_classes = getattr(model, 'num_classes', len(class_names))
        reg_max = getattr(model, 'reg_max', 16) # Get reg_max from model
        k_model = getattr(model, 'max_objects_per_pixel', max_objects_per_pixel) # Get K from model
        if k_model != max_objects_per_pixel:
            print(f"Warning: K from model ({k_model}) != K specified ({max_objects_per_pixel}). Using value from model.")
            max_objects_per_pixel = k_model
        # --- Adjust class names based on model's num_classes ---
        if num_classes != len(class_names):
             print(f"Warning: Model num_classes ({num_classes}) differs from len(class_names) ({len(class_names)}). Adjusting class_names list.")
             if num_classes < len(class_names): class_names = class_names[:num_classes]
             else: class_names.extend([f"cls_{i}" for i in range(len(class_names), num_classes)])
    except Exception as e: print(f"Error loading model: {e}"); traceback.print_exc(); return None
    print(f"Model loaded to {eval_device}. Classes: {num_classes}, RegMax: {reg_max}, K: {max_objects_per_pixel}")


    # --- 2. Initialize Metric ---
    metric = torchmetrics.detection.MeanAveragePrecision(
        iou_type="bbox", class_metrics=True
    ).to(eval_device)

    # --- 3. Pre-calculate Anchors/Strides (assuming constant image size) ---
    print(f"Generating anchors/strides for image size {img_size}x{img_size}...")
    anchor_points_norm, strides_map = generate_anchor_points_and_strides(
        strides=strides, img_h=img_size, img_w=img_size, k=max_objects_per_pixel, device=eval_device
    )
    if anchor_points_norm.shape[0] == 0:
        print("Error: Failed to generate anchor points. Check strides/img_size.")
        return None

    # --- 4. Evaluation Loop ---
    print(f"Starting evaluation on '{dataset_name}' dataset...")
    loop = tqdm(data_loader, desc=f"Evaluating {dataset_name}")

    for batch_idx, batch_data in enumerate(loop):
        try:
            images, _, raw_targets_list = batch_data # Ignore formatted targets for loss
        except Exception as e: print(f"Batch {batch_idx} unpack error: {e}. Skipping."); continue

        if images is None or len(images) == 0: continue
        images = images.to(eval_device)
        current_img_h, current_img_w = images.shape[2:]

        # --- Check if image size changed (if assuming constant) ---
        if current_img_h != img_size or current_img_w != img_size:
             print(f"Warning: Batch {batch_idx} img size ({current_img_h},{current_img_w}) differs from assumed ({img_size},{img_size}). Recalculating anchors...")
             anchor_points_norm_batch, strides_map_batch = generate_anchor_points_and_strides(
                 strides=strides, img_h=current_img_h, img_w=current_img_w, k=max_objects_per_pixel, device=eval_device
             )
             if anchor_points_norm_batch.shape[0] == 0: print("Anchor gen failed for batch. Skipping."); continue
        else:
             anchor_points_norm_batch, strides_map_batch = anchor_points_norm, strides_map

        try:
            with torch.no_grad():
                outputs = model(images) # Get raw model outputs

                # Decode predictions (DFL version)
                decoded_preds = decode_dfl_predictions(
                    outputs, anchor_points_norm_batch, strides_map_batch,
                    reg_max, current_img_h, current_img_w
                )

                # Post-process (Conf threshold, NMS)
                # Output boxes are absolute pixels [x1, y1, x2, y2]
                final_preds = post_process_batch(decoded_preds, conf_thres, iou_thres, min_box_area)

                # Format ground truth targets for torchmetrics (absolute pixels [x1, y1, x2, y2])
                formatted_targets = format_targets_for_metric(raw_targets_list, current_img_h, current_img_w, device=eval_device)

                # Update metric
                metric.update(final_preds, formatted_targets)

        except Exception as e_inner:
            print(f"\nError during batch {batch_idx} processing: {e_inner}")
            traceback.print_exc(); print("Skipping batch.")

    # --- 5. Compute and Print Results ---
    print("\nEvaluation finished. Computing final metrics...")
    try:
        results = metric.compute()
    except Exception as e: print(f"Error computing metrics: {e}"); traceback.print_exc(); return None

    print("\n--- Evaluation Results ---")
    print(f"Model: {model_path}, Dataset: {dataset_name}")
    print(f"Conf: {conf_thres}, NMS IoU: {iou_thres}, Min Area: {min_box_area}")
    print("-" * 25)
    print("[Overall Metrics]")
    map_50_95 = results.get('map', torch.tensor(-1.0)).item()
    map_50 = results.get('map_50', torch.tensor(-1.0)).item()
    map_75 = results.get('map_75', torch.tensor(-1.0)).item()
    mar_small = results.get('mar_small', torch.tensor(-1.0)).item()
    mar_medium = results.get('mar_medium', torch.tensor(-1.0)).item()
    mar_large = results.get('mar_large', torch.tensor(-1.0)).item()
    print(f"  mAP@.5:.95 = {map_50_95:.4f}")
    print(f"  mAP@.50   = {map_50:.4f}")
    print(f"  mAP@.75   = {map_75:.4f}")
    print(f"  Recall[L] = {mar_large:.4f}")
    print(f"  Recall[M] = {mar_medium:.4f}")
    print(f"  Recall[S] = {mar_small:.4f}")
    print("-" * 25)
    print("[Per-Class mAP@.5:.95]")
    map_per_class = results.get('map_per_class', torch.tensor([])); classes = results.get('classes', torch.tensor([]))
    recall_key = 'mar_large_per_class'; recall_per_class = results.get(recall_key, torch.tensor([]))

    if map_per_class.numel() > 0 and classes.numel() > 0 and map_per_class.shape == classes.shape:
        class_maps = {int(c.item()): m.item() for c, m in zip(classes, map_per_class) if m >= 0}
        class_recalls = {}
        if recall_per_class.numel() > 0 and recall_per_class.shape == classes.shape:
             class_recalls = {int(c.item()): r.item() for c, r in zip(classes, recall_per_class) if r >= 0}

        print(f"{'Class':<20} {'mAP':<10} {'Recall[L]':<10}")
        print("-" * 45)
        for i, name in enumerate(class_names):
            map_val = class_maps.get(i, -1.0); recall_val = class_recalls.get(i, -1.0)
            if map_val != -1.0 : print(f"  {name:<18} {map_val:<10.4f} {recall_val:<10.4f}")
    else: print("Could not print per-class results.")
    print("-" * 25)
    return results


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == "__main__":
    # --- Import necessary components ---
    from config.config import CLASS_NAMES as CLASS_NAMES_LIST
    from src.detection_model_architecture import DetectionModel
    from src.dataset_handlers.full_coco_dataset_handler import CocoDataset
    from src.save_tools import load_model
    from src.collate_fn import collate_fn

    # --- Configuration ---
    MODEL_PATH = "train/train_149/best.pt" 
    DATASET_FOLDER = 'datasets/aquarium' 
    IMG_SIZE = 640
    BATCH_SIZE = 8 # Adjust based on GPU memory for evaluation
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    STRIDES = [8, 16, 32] # Should match model
    MAX_OBJECTS_PER_PIXEL = 1 # Should match model (K value)

    # --- Eval Transform (No Augmentations, just resize/pad) ---
    def get_eval_augment_transform(target_size=640):
         return A.Compose([
             A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LINEAR),
             A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), position='center', p=1.0),
         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # --- Create Validation DataLoader ---
    print("Setting up Validation dataloader for evaluation...")
    eval_val_loader = None
    try:
        eval_dataset = CocoDataset(
            coco_folder=DATASET_FOLDER,
            val=True,
            augment_transform=get_eval_augment_transform(IMG_SIZE),
            target_size=IMG_SIZE,
            enable_mosaic=False 
        )
        eval_val_loader = DataLoader(
            eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, 
            pin_memory=True if DEVICE == 'cuda' else False,
            num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1),
            persistent_workers=False 
        )
        print("Validation Dataloader ready.")
    except Exception as e:
        print(f"Error setting up evaluation dataloader: {e}"); traceback.print_exc()

    # --- Run Evaluation ---
    if eval_val_loader:
        evaluate_model(
            model_path=MODEL_PATH,
            model_type=DetectionModel, 
            data_loader=eval_val_loader, 
            dataset_name="Validation",
            class_names=CLASS_NAMES_LIST,
            device=DEVICE,
            img_size=IMG_SIZE,
            strides=STRIDES,
            max_objects_per_pixel=MAX_OBJECTS_PER_PIXEL,
            conf_thres=0.001,
            iou_thres=0.7, 
        )
    else:
        print("\nSkipping evaluation as dataloader setup failed.")

