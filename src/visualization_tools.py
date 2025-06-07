# visualization_tools.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision
import os
import torch.nn.functional as F 
import pandas as pd

from src.config.config import CLASS_NAMES



def box_iou(boxes1_xyxy, boxes2_xyxy):
    boxes1_xyxy = torch.as_tensor(boxes1_xyxy, dtype=torch.float32)
    boxes2_xyxy = torch.as_tensor(boxes2_xyxy, dtype=torch.float32)
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - intersection + 1e-6
    return intersection / union

def find_next_train_folder(base_dir="train/train"):
    k = 1
    while True:
        folder_name = f"{base_dir}_{k}"
        if not os.path.exists(folder_name):
            return folder_name, k
        k += 1

def denormalize_image(tensor):
    tensor = tensor.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def cxcywh_to_xyxy_torch(box_cxcywh):
    if not isinstance(box_cxcywh, torch.Tensor):
        box_cxcywh = torch.as_tensor(box_cxcywh, dtype=torch.float32)
    if box_cxcywh.numel() == 0:
        return torch.empty((0, 4), dtype=box_cxcywh.dtype, device=box_cxcywh.device)
    
    original_ndim = box_cxcywh.ndim # Store original ndim
    if original_ndim == 1: # If input is 1D, e.g. shape (4,)
        box_cxcywh = box_cxcywh.unsqueeze(0) # Make it 2D: shape (1, 4)
    
    # Now, box_cxcywh is guaranteed to be 2D, e.g., (N, 4) or (1, 4)
    # Ensure last dimension is 4
    if box_cxcywh.shape[-1] != 4:
        raise ValueError(f"Input box_cxcywh expected to have 4 elements in the last dimension, got {box_cxcywh.shape}")

    cx, cy, w, h = box_cxcywh.unbind(-1) # Unbinds along the last dimension
                                         # If input was (N,4), cx is (N,). If (1,4), cx is (1,).
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    output = torch.stack([x1, y1, x2, y2], dim=-1) # If input was (N,4), output is (N,4). If (1,4), output is (1,4).

    if original_ndim == 1: # If original input was a single 1D box (e.g., shape (4,))
        return output.squeeze(0) # Squeeze back to 1D (shape (4,))
    else: # If original input was already 2D (e.g., shape (N,4) where N could be 1)
        return output # Return as 2D (shape (N,4))

# --- Updated Main Visualization Function ---
def visualize_training_batch(model, images, targets, class_names, device, epoch, batch_idx, save_folder, max_images=8):
    model.eval()
    K = getattr(model, 'max_objects_per_pixel', 1)
    reg_max = getattr(model, 'reg_max', 16) # Default to 16 if not found

    with torch.no_grad():
        model.to(device)
        predictions_from_model = model(images.to(device))

    images = images.cpu()
    cpu_targets = []
    if targets is not None:
        for t_dict in targets:
            cpu_targets.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t_dict.items()})
    targets = cpu_targets

    predictions = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in predictions_from_model.items()}
    pred_device = torch.device('cpu') # For local tensor creations

    os.makedirs(save_folder, exist_ok=True)

    OBJ_THRESHOLD = 0.3
    CLS_THRESHOLD = 0.3
    CONF_THRESHOLD = 0.3 # Final confidence (obj * cls)
    MIN_AREA_PIXELS = 10
    NMS_IOU_THRESHOLD = 0.4
    MAX_FINAL_DETECTIONS = 100
    
    strides = getattr(model, 'strides', [8, 16, 32])
    project_tensor = torch.arange(reg_max, device=pred_device, dtype=torch.float32)


    for i in range(min(max_images, len(images))):
        img_tensor = images[i]
        img_h_in, img_w_in = img_tensor.shape[1:] # Model input H, W
        img_to_display_normalized = denormalize_image(img_tensor)
        img_np_to_display = img_to_display_normalized.permute(1, 2, 0).numpy()
        img_h_draw, img_w_draw = img_np_to_display.shape[:2] # Actual pixel dims for drawing

        target = targets[i] if targets and i < len(targets) else {}

        # --- Get predictions for the current image ---
        # Output of DFL head is logits for l,t,r,b distributions
        pred_dfl_logits_img = predictions['boxes_dfl'][i]      # Shape (N_total, 4 * reg_max)
        pred_classes_logits_img = predictions['classes'][i]    # Shape (N_total, C)
        pred_objectness_logits_img = predictions['objectness'][i].squeeze(-1) # Shape (N_total,)
        
        N_total_preds_from_model = pred_dfl_logits_img.shape[0]

        fig, ax = plt.subplots(1, figsize=(12, 9))
        fig.suptitle(f"Epoch {epoch} | Batch {batch_idx} | Sample {i}", fontsize=14)
        ax.imshow(img_np_to_display)

        # --- Draw Ground Truth ---
        gt_boxes_xyxy_for_iou = []
        gt_labels_for_iou = []
        if 'boxes' in target and 'labels' in target and len(target['boxes']) > 0:
            gt_boxes_cxcywh_cpu = torch.as_tensor(target['boxes'], device=pred_device)
            gt_labels_cpu = torch.as_tensor(target['labels'], device=pred_device)
            for box_idx_gt in range(gt_boxes_cxcywh_cpu.shape[0]):
                box_cxcywh_rel_gt = gt_boxes_cxcywh_cpu[box_idx_gt]
                gt_box_xyxy_rel = cxcywh_to_xyxy_torch(box_cxcywh_rel_gt)
                gt_boxes_xyxy_for_iou.append(gt_box_xyxy_rel)
                gt_labels_for_iou.append(gt_labels_cpu[box_idx_gt].item())

                cx, cy, w, h = box_cxcywh_rel_gt.numpy()
                x1_abs = (cx - w / 2) * img_w_draw; y1_abs = (cy - h / 2) * img_h_draw
                w_abs = w * img_w_draw; h_abs = h * img_h_draw
                rect = patches.Rectangle((x1_abs, y1_abs), w_abs, h_abs, lw=2, edgecolor='r', facecolor='none', ls='--')
                ax.add_patch(rect)
                label_idx = gt_labels_cpu[box_idx_gt].item()
                cls_name = class_names[label_idx] if class_names and 0<=label_idx<len(class_names) else f"ID:{label_idx}"
                ax.text(x1_abs, y1_abs - 5, f"GT: {cls_name}", color='w', fontsize=8, bbox=dict(facecolor='r', alpha=0.6, pad=1, edgecolor='none'))
        
        if gt_boxes_xyxy_for_iou:
            gt_boxes_xyxy_for_iou_tensor = torch.stack(gt_boxes_xyxy_for_iou).to(pred_device)
            gt_labels_for_iou_tensor = torch.tensor(gt_labels_for_iou, dtype=torch.long, device=pred_device)
        else:
            gt_boxes_xyxy_for_iou_tensor = torch.empty((0, 4), device=pred_device)
            gt_labels_for_iou_tensor = torch.empty((0,), dtype=torch.long, device=pred_device)

        # --- Decode DFL Predictions ---
        # 1. Reconstruct anchor points (normalized cell centers) and strides
        anchor_points_norm_list = []
        strides_list_for_decode = []
        for stride_val in strides:
            H_grid, W_grid = img_h_in // stride_val, img_w_in // stride_val
            if H_grid > 0 and W_grid > 0:
                num_cells_on_scale = H_grid * W_grid
                gy, gx = torch.meshgrid(torch.arange(H_grid,device=pred_device), torch.arange(W_grid,device=pred_device), indexing='ij')
                anchors_scale = torch.stack([(gx.flatten()+0.5)/W_grid, (gy.flatten()+0.5)/H_grid], dim=-1).float()
                strides_scale = torch.full((num_cells_on_scale, 1), stride_val, device=pred_device, dtype=torch.float32)
                anchor_points_norm_list.append(anchors_scale.repeat_interleave(K, dim=0))
                strides_list_for_decode.append(strides_scale.repeat_interleave(K, dim=0))

        if not anchor_points_norm_list: # No valid anchors generated
            decoded_boxes_flat_xyxy_rel = torch.empty((0, 4), device=pred_device)
            N_total_anchors_generated = 0
        else:
            anchor_points_norm_viz_flat = torch.cat(anchor_points_norm_list, dim=0) # (N_total_anchors, 2)
            strides_viz_flat = torch.cat(strides_list_for_decode, dim=0)      # (N_total_anchors, 1)
            N_total_anchors_generated = anchor_points_norm_viz_flat.shape[0]

        # Align number of predictions with generated anchors
        if N_total_anchors_generated != N_total_preds_from_model:
            min_n = min(N_total_anchors_generated, N_total_preds_from_model)
            # print(f"Viz Warning: Anchor/Pred mismatch. Anchors:{N_total_anchors_generated}, Preds:{N_total_preds_from_model}. Using {min_n}")
            anchor_points_norm_viz_flat = anchor_points_norm_viz_flat[:min_n]
            strides_viz_flat = strides_viz_flat[:min_n]
            pred_dfl_logits_img = pred_dfl_logits_img[:min_n]
            pred_classes_logits_img = pred_classes_logits_img[:min_n]
            pred_objectness_logits_img = pred_objectness_logits_img[:min_n]
            N_total_effective = min_n
        else:
            N_total_effective = N_total_anchors_generated
            
        if N_total_effective == 0: # No predictions to process after alignment
            ax.axis('off'); plt.tight_layout(pad=0.1)
            filename = os.path.join(save_folder, f"e{epoch}_b{batch_idx}_s{i}_nopreds.png")
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1); plt.close(fig)
            continue # next image in batch

        # 2. Decode DFL logits to LTRB distances (in stride units)
        # pred_dfl_logits_img is (N_total_effective, 4 * reg_max)
        reshaped_dfl = pred_dfl_logits_img.view(N_total_effective, 4, reg_max)
        distributions = F.softmax(reshaped_dfl, dim=2) # (N_total_effective, 4, reg_max)
        pred_ltrb_stride_units = distributions.matmul(project_tensor) # (N_total_effective, 4)

        # 3. Convert LTRB (stride units) to XYXY (normalized image coordinates)
        s = strides_viz_flat.squeeze(-1) # (N_total_effective,)
        anchor_cx_norm = anchor_points_norm_viz_flat[:, 0]
        anchor_cy_norm = anchor_points_norm_viz_flat[:, 1]

        # Distances in pixel units: pred_ltrb_stride_units * s_expanded
        # Distances in normalized units: (pred_ltrb_stride_units * s_expanded) / image_input_dim
        pred_x1_norm = anchor_cx_norm - (pred_ltrb_stride_units[:, 0] * s) / img_w_in
        pred_y1_norm = anchor_cy_norm - (pred_ltrb_stride_units[:, 1] * s) / img_h_in
        pred_x2_norm = anchor_cx_norm + (pred_ltrb_stride_units[:, 2] * s) / img_w_in
        pred_y2_norm = anchor_cy_norm + (pred_ltrb_stride_units[:, 3] * s) / img_h_in
        decoded_boxes_flat_xyxy_rel = torch.stack([pred_x1_norm, pred_y1_norm, pred_x2_norm, pred_y2_norm], dim=-1)
        decoded_boxes_flat_xyxy_rel = decoded_boxes_flat_xyxy_rel.clamp(min=0.0, max=1.0) # Clamp to [0,1]

        # --- Score Filtering & NMS (operates on XYXY boxes) ---
        obj_probs = torch.sigmoid(pred_objectness_logits_img)
        cls_probs = torch.sigmoid(pred_classes_logits_img) # Assuming BCE loss for classes
        max_cls_probs, max_cls_indices = torch.max(cls_probs, dim=1)
        confidence_scores = obj_probs * max_cls_probs

        # Area mask
        box_w_rel = (decoded_boxes_flat_xyxy_rel[:, 2] - decoded_boxes_flat_xyxy_rel[:, 0]).clamp(min=0)
        box_h_rel = (decoded_boxes_flat_xyxy_rel[:, 3] - decoded_boxes_flat_xyxy_rel[:, 1]).clamp(min=0)
        area_abs_pixels = (box_w_rel * img_w_draw) * (box_h_rel * img_h_draw) # Use drawing dims for area
        area_mask = (area_abs_pixels >= MIN_AREA_PIXELS) & (box_w_rel > 0) & (box_h_rel > 0)
        
        keep_mask = (confidence_scores >= CONF_THRESHOLD) & \
                    area_mask & \
                    (obj_probs >= OBJ_THRESHOLD) & \
                    (max_cls_probs >= CLS_THRESHOLD)
        
        filtered_indices = torch.where(keep_mask)[0]
        
        final_pred_boxes_xyxy = torch.empty((0,4), device=pred_device)
        final_pred_labels = torch.empty((0,), dtype=torch.long, device=pred_device)
        final_pred_scores = torch.empty((0,), device=pred_device)
        final_pred_best_ious = torch.empty((0,), device=pred_device)

        if filtered_indices.numel() > 0:
            filt_boxes_xyxy = decoded_boxes_flat_xyxy_rel[filtered_indices]
            filt_labels = max_cls_indices[filtered_indices]
            filt_confidences = confidence_scores[filtered_indices] # Original confidence

            # Enhanced score with IoU (optional, can use filt_confidences directly for NMS)
            filt_best_ious = torch.zeros(filtered_indices.numel(), device=pred_device)
            if gt_boxes_xyxy_for_iou_tensor.numel() > 0:
                iou_matrix = box_iou(filt_boxes_xyxy, gt_boxes_xyxy_for_iou_tensor) # (N_filt, N_gt)
                for j_pred_idx in range(filt_boxes_xyxy.shape[0]):
                    pred_cls = filt_labels[j_pred_idx]
                    gt_matching_cls_mask = (gt_labels_for_iou_tensor == pred_cls)
                    if gt_matching_cls_mask.any():
                        ious_with_same_class = iou_matrix[j_pred_idx, gt_matching_cls_mask]
                        if ious_with_same_class.numel() > 0:
                            filt_best_ious[j_pred_idx] = ious_with_same_class.max()
            
            # Using original confidence for NMS, not enhanced score for simplicity here
            scores_for_nms = filt_confidences 

            # Per-class NMS
            unique_pred_labels = torch.unique(filt_labels)
            nms_keep_indices = []
            for class_id_val in unique_pred_labels:
                class_mask = (filt_labels == class_id_val)
                class_boxes_for_nms = filt_boxes_xyxy[class_mask]
                class_scores_for_nms = scores_for_nms[class_mask]
                
                # torchvision.ops.nms returns indices relative to the input of this specific NMS call
                kept_in_class_indices = torchvision.ops.nms(class_boxes_for_nms, class_scores_for_nms, NMS_IOU_THRESHOLD)
                
                # Map back to indices in 'filtered_indices'
                original_indices_of_class_preds = torch.where(class_mask)[0]
                nms_keep_indices.extend(original_indices_of_class_preds[kept_in_class_indices].tolist())

            if nms_keep_indices:
                # These are indices into the `filtered_indices` array
                final_indices_in_filtered_array = torch.tensor(nms_keep_indices, dtype=torch.long, device=pred_device)
                
                final_pred_boxes_xyxy = filt_boxes_xyxy[final_indices_in_filtered_array]
                final_pred_labels = filt_labels[final_indices_in_filtered_array]
                final_pred_scores = filt_confidences[final_indices_in_filtered_array] # Original confidence
                final_pred_best_ious = filt_best_ious[final_indices_in_filtered_array]

                # Limit to MAX_FINAL_DETECTIONS based on score
                if final_pred_scores.numel() > MAX_FINAL_DETECTIONS:
                    _, sort_order = torch.sort(final_pred_scores, descending=True)
                    top_k_post_nms = sort_order[:MAX_FINAL_DETECTIONS]
                    final_pred_boxes_xyxy = final_pred_boxes_xyxy[top_k_post_nms]
                    final_pred_labels = final_pred_labels[top_k_post_nms]
                    final_pred_scores = final_pred_scores[top_k_post_nms]
                    final_pred_best_ious = final_pred_best_ious[top_k_post_nms]
        
        # --- Draw Final Predictions ---
        for k_final_idx in range(final_pred_boxes_xyxy.shape[0]):
            box_xyxy_rel = final_pred_boxes_xyxy[k_final_idx]
            label_idx = final_pred_labels[k_final_idx].item()
            score = final_pred_scores[k_final_idx].item()
            best_iou_val = final_pred_best_ious[k_final_idx].item()

            cls_name = class_names[label_idx] if class_names and 0<=label_idx<len(class_names) else f"ID:{label_idx}"

            x1_abs = box_xyxy_rel[0].item() * img_w_draw
            y1_abs = box_xyxy_rel[1].item() * img_h_draw
            x2_abs = box_xyxy_rel[2].item() * img_w_draw
            y2_abs = box_xyxy_rel[3].item() * img_h_draw
            
            w_abs = x2_abs - x1_abs
            h_abs = y2_abs - y1_abs

            if w_abs > 0 and h_abs > 0: # Only draw if valid box
                rect = patches.Rectangle((x1_abs,y1_abs),w_abs,h_abs,lw=1.5,edgecolor='lime',facecolor='none')
                ax.add_patch(rect)
                label_txt = f"{cls_name} {score:.2f} (IoU:{best_iou_val:.2f})"
                ax.text(x1_abs, y1_abs - 5, label_txt, color='k', fontsize=7,
                        bbox=dict(facecolor='lime', alpha=0.7, pad=1, edgecolor='none'))

        ax.axis('off'); plt.tight_layout(pad=0.1)
        filename = os.path.join(save_folder, f"e{epoch}_b{batch_idx}_s{i}_viz.png")
        try:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        except Exception as e_save:
            print(f"Error saving plot {filename}: {e_save}")
        plt.close(fig)
    model.train()


def plot_losses_from_csv(csv_path, output_folder=None, smoothing_window=10):
    """Generate separate, clean loss plots for each component from CSV data."""
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"CSV file '{csv_path}' is empty. No plots will be generated.")
            return
    except pd.errors.EmptyDataError:
        print(f"CSV file '{csv_path}' is empty or unreadable. No plots will be generated.")
        return
    except FileNotFoundError:
        print(f"CSV file '{csv_path}' not found. No plots will be generated.")
        return
    except Exception as e:
        print(f"Error reading CSV '{csv_path}': {e}")
        return

    # Prepare data
    # Ensure 'type' column exists
    if 'type' not in df.columns:
        print("CSV missing 'type' column. Cannot differentiate data.")
        return
        
    train_batch = df[df['type'] == 'train_batch'].copy()
    train_epoch = df[df['type'] == 'train_epoch'].copy()
    val_batch = df[df['type'] == 'val_batch'].copy()
    val_epoch = df[df['type'] == 'val_epoch'].copy()

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    colors = {'total': '#1f77b4', 'box': '#ff7f0e', 'cls': '#2ca02c', 'obj': '#d62728'}
    val_color = '#9467bd'
    
    loss_types_to_plot = []
    for lt in ['total', 'box', 'cls', 'obj']:
        if f"{lt}_loss" in df.columns:
            loss_types_to_plot.append(lt)
        else:
            print(f"Warning: Column '{lt}_loss' not found in CSV. Skipping plot for this loss type.")

    if not loss_types_to_plot:
        print("No valid loss columns found to plot (e.g., 'total_loss', 'box_loss').")
        return

    for loss_type in loss_types_to_plot:
        col_name = f"{loss_type}_loss"
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18)) # Increased height
        fig.suptitle(f"{loss_type.upper()} Loss Analysis", y=0.99, fontsize=16) 
        
        # Plot 1: Batch-level training loss
        ax1_has_data = False
        if not train_batch.empty and col_name in train_batch.columns:
            plot_x_ax1 = np.arange(len(train_batch)) 
            
            # Ensure data is numeric and drop NaNs before rolling/min/max
            train_batch_col_data = pd.to_numeric(train_batch[col_name], errors='coerce').dropna()

            if not train_batch_col_data.empty:
                smoothed = train_batch_col_data.rolling(window=smoothing_window, min_periods=1).mean()
                
                ax1.plot(plot_x_ax1[:len(train_batch_col_data)], train_batch_col_data, 
                         color=colors[loss_type], alpha=0.25, linewidth=1, label='Raw Batch')
                ax1.plot(plot_x_ax1[:len(smoothed)], smoothed, 
                         color=colors[loss_type], linewidth=2, label=f'Smoothed (w={smoothing_window})')
                ax1_has_data = True

                y_min_ax1, y_max_ax1 = smoothed.min(), smoothed.max() # Smoothed data for cleaner limits
                if pd.notna(y_min_ax1) and pd.notna(y_max_ax1): # Check if min/max are valid numbers
                    padding = (y_max_ax1 - y_min_ax1) * 0.1 if (y_max_ax1 - y_min_ax1) > 1e-5 else 0.1
                    ax1.set_ylim(max(0, y_min_ax1 - padding), y_max_ax1 + padding)

        ax1.set_xlabel('Batch Sequence Index')
        ax1.set_ylabel('Loss Value')
        ax1.set_title(f'Training Batch {loss_type.upper()} Loss')
        ax1.grid(True, alpha=0.3)
        if ax1_has_data: ax1.legend()
        else: ax1.text(0.5, 0.5, 'No batch training data for this loss', ha='center', va='center', transform=ax1.transAxes); ax1.set_xticks([]); ax1.set_yticks([])

        # Plot 2: Epoch-level training and validation
        ax2_has_data = False
        all_ax2_plot_values = []

        if not train_epoch.empty and 'epoch' in train_epoch.columns and col_name in train_epoch.columns:
            train_epoch_col_data = pd.to_numeric(train_epoch[col_name], errors='coerce').dropna()
            if not train_epoch_col_data.empty:
                # Align epoch_x with potentially dropped NaN values
                valid_train_indices = train_epoch_col_data.index
                ax2.plot(train_epoch['epoch'].loc[valid_train_indices], train_epoch_col_data, 
                         'o-', color=colors[loss_type], label='Train Epoch Avg')
                all_ax2_plot_values.extend(train_epoch_col_data.tolist())
                ax2_has_data = True
            
        if not val_epoch.empty and 'epoch' in val_epoch.columns and col_name in val_epoch.columns:
            val_epoch_col_data = pd.to_numeric(val_epoch[col_name], errors='coerce').dropna()
            if not val_epoch_col_data.empty:
                valid_val_indices = val_epoch_col_data.index
                ax2.plot(val_epoch['epoch'].loc[valid_val_indices], val_epoch_col_data, 
                         's--', color=val_color, label='Validation Epoch Avg')
                all_ax2_plot_values.extend(val_epoch_col_data.tolist())
                ax2_has_data = True
            
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Value')
        ax2.set_title(f'Epoch {loss_type.upper()} Loss')
        ax2.grid(True, alpha=0.3)

        if ax2_has_data:
            ax2.legend()
            if all_ax2_plot_values: # Ensure list is not empty
                min_val_ax2 = min(all_ax2_plot_values)
                max_val_ax2 = max(all_ax2_plot_values)
                if pd.notna(min_val_ax2) and pd.notna(max_val_ax2): # Check if min/max are valid numbers
                    if max_val_ax2 > min_val_ax2:
                        padding = (max_val_ax2 - min_val_ax2) * 0.1 if (max_val_ax2 - min_val_ax2) > 1e-5 else 0.1
                        ax2.set_ylim(max(0, min_val_ax2 - padding), max_val_ax2 + padding)
                    elif max_val_ax2 == min_val_ax2: # Single point or all same value
                        padding = abs(max_val_ax2 * 0.1) if max_val_ax2 != 0 else 0.1
                        ax2.set_ylim(max_val_ax2 - padding, max_val_ax2 + padding)
        else:
            ax2.text(0.5, 0.5, 'No epoch data for this loss', ha='center', va='center', transform=ax2.transAxes); ax2.set_xticks([]); ax2.set_yticks([])


        # Plot 3: Validation batch points (if available)
        ax3_has_data = False
        if not val_batch.empty and col_name in val_batch.columns:
            plot_x_ax3 = np.arange(len(val_batch)) # Simple sequential index
            val_batch_col_data = pd.to_numeric(val_batch[col_name], errors='coerce').dropna()

            if not val_batch_col_data.empty:
                val_smoothed = val_batch_col_data.rolling(window=smoothing_window, min_periods=1).mean()
                
                ax3.plot(plot_x_ax3[:len(val_batch_col_data)], val_batch_col_data, 
                         'o', alpha=0.3, color=val_color, markersize=3, label='Val Batch Raw')
                ax3.plot(plot_x_ax3[:len(val_smoothed)], val_smoothed, 
                         '-', alpha=0.7, color=val_color, linewidth=1.5, label=f'Val Batch Smoothed (w={smoothing_window})')
                ax3_has_data = True

                y_min_ax3, y_max_ax3 = val_smoothed.min(), val_smoothed.max()
                if pd.notna(y_min_ax3) and pd.notna(y_max_ax3):
                    padding = (y_max_ax3 - y_min_ax3) * 0.1 if (y_max_ax3 - y_min_ax3) > 1e-5 else 0.1
                    ax3.set_ylim(max(0, y_min_ax3 - padding), y_max_ax3 + padding)
        
        ax3.set_xlabel('Validation Batch Sequence Index')
        ax3.set_ylabel('Loss Value')
        ax3.set_title(f'Validation Batch {loss_type.upper()} Loss')
        ax3.grid(True, alpha=0.3)
        if ax3_has_data: ax3.legend()
        else: ax3.text(0.5, 0.5, 'No validation batch data for this loss', ha='center', va='center', transform=ax3.transAxes); ax3.set_xticks([]); ax3.set_yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
        
        if output_folder:
            plot_path = os.path.join(output_folder, f"{loss_type}_loss_analysis.png") # More descriptive name
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            plt.close(fig) 
        else:
            plt.show() 

    # --- Plot for Loss Component Ratios ---
    if not train_epoch.empty and 'total_loss' in train_epoch.columns and \
       all(f"{comp}_loss" in train_epoch.columns for comp in ['box', 'cls', 'obj']):
        
        total_loss_series = pd.to_numeric(train_epoch['total_loss'], errors='coerce')
        total_loss_series_safe = total_loss_series.replace(0, 1e-9).fillna(1e-9)


        fig_ratio, ax_ratio = plt.subplots(figsize=(12, 6))
        has_ratio_data = False
        for component in ['box', 'cls', 'obj']:
            comp_col_name = f"{component}_loss"
            comp_loss_series = pd.to_numeric(train_epoch[comp_col_name], errors='coerce')
            
            valid_indices = total_loss_series.index.intersection(comp_loss_series.index)
            if not valid_indices.empty:
                ratios = comp_loss_series.loc[valid_indices] / total_loss_series_safe.loc[valid_indices]
                ratios = ratios.dropna() # Remove any NaNs from division
                if not ratios.empty:
                    ax_ratio.plot(train_epoch['epoch'].loc[ratios.index], ratios, 
                                  'o-', color=colors[component], label=f"{component.upper()} Ratio")
                    has_ratio_data = True
        
        if has_ratio_data:
            ax_ratio.set_xlabel('Epoch')
            ax_ratio.set_ylabel('Ratio to Total Loss')
            ax_ratio.set_title('Training Loss Component Ratios (Epoch Avg)')
            ax_ratio.set_ylim(0, 1) # Ratios should be between 0 and 1
            ax_ratio.grid(True, alpha=0.3)
            ax_ratio.legend()
            plt.tight_layout()

            if output_folder:
                plot_path_ratio = os.path.join(output_folder, "loss_component_ratios.png")
                plt.savefig(plot_path_ratio, bbox_inches='tight', dpi=150)
                plt.close(fig_ratio)
            else:
                plt.show()
        else:
            print("Not enough data to plot loss component ratios.")
            plt.close(fig_ratio) 
    else:
        print("Skipping loss component ratios plot: Missing necessary loss columns in train_epoch data.")






def update_loss_curves(epoch_losses, recent_batch_losses, save_dir, epoch, batch_idx):
    """Update and save comprehensive loss curves with all components."""
    plt.figure(figsize=(24, 16))
    
    # Plot 1: Training Loss Components per Epoch
    plt.subplot(3, 3, 1)
    epochs_range = range(1, len(epoch_losses['train']['total']) + 1)
    plt.plot(epochs_range, epoch_losses['train']['total'], label='Total Loss')
    plt.plot(epochs_range, epoch_losses['train']['box'], label='Box Loss')
    plt.plot(epochs_range, epoch_losses['train']['cls'], label='Cls Loss')
    plt.plot(epochs_range, epoch_losses['train']['obj'], label='Obj Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components (Epoch Level)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Recent Batch Loss Components
    plt.subplot(3, 3, 2)
    batch_range = range(len(recent_batch_losses['total']))
    plt.plot(batch_range, recent_batch_losses['total'], label='Total', alpha=0.7)
    plt.plot(batch_range, recent_batch_losses['box'], label='Box', alpha=0.7)
    plt.plot(batch_range, recent_batch_losses['cls'], label='Cls', alpha=0.7)
    plt.plot(batch_range, recent_batch_losses['obj'], label='Obj', alpha=0.7)
    plt.xlabel('Recent Batch Index')
    plt.ylabel('Loss')
    plt.title(f'Recent {len(batch_range)} Batch Loss Components')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Smoothed Batch Loss Components
    plt.subplot(3, 3, 3)
    window_size = max(5, len(recent_batch_losses['total']) // 20)
    for loss_type in ['total', 'box', 'cls', 'obj']:
        smoothed = [sum(recent_batch_losses[loss_type][i:i+window_size])/window_size 
                   for i in range(len(recent_batch_losses[loss_type])-window_size)]
        plt.plot(range(window_size, len(recent_batch_losses[loss_type])), 
                smoothed, label=f'{loss_type} (window={window_size})')
    plt.xlabel('Recent Batch Index')
    plt.ylabel('Smoothed Loss')
    plt.title('Smoothed Batch Loss Components')
    plt.grid(True)
    plt.legend()
    
    # Plot 4-6: Validation Loss Components (if available)
    if epoch_losses['val']['total']:
        val_epochs = [i * (len(epochs_range)/len(epoch_losses['val']['total'])) 
                     for i in range(len(epoch_losses['val']['total']))]
        
        plt.subplot(3, 3, 4)
        plt.plot(val_epochs, epoch_losses['val']['total'], 'o-', label='Total')
        plt.plot(val_epochs, epoch_losses['val']['box'], 'o-', label='Box')
        plt.plot(val_epochs, epoch_losses['val']['cls'], 'o-', label='Cls')
        plt.plot(val_epochs, epoch_losses['val']['obj'], 'o-', label='Obj')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Components')
        plt.grid(True)
        plt.legend()
        
        # Plot 5: Validation vs Training Comparison (Total Loss)
        plt.subplot(3, 3, 5)
        plt.plot(epochs_range, epoch_losses['train']['total'], label='Train Total')
        plt.plot(val_epochs, epoch_losses['val']['total'], 'o-', label='Val Total')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation (Total Loss)')
        plt.grid(True)
        plt.legend()
        
        # Plot 6: Validation vs Training (Component Ratios)
        plt.subplot(3, 3, 6)
        width = 0.2
        train_ratios = [
            sum(epoch_losses['train']['box']) / sum(epoch_losses['train']['total']),
            sum(epoch_losses['train']['cls']) / sum(epoch_losses['train']['total']),
            sum(epoch_losses['train']['obj']) / sum(epoch_losses['train']['total'])
        ]
        val_ratios = [
            sum(epoch_losses['val']['box']) / sum(epoch_losses['val']['total']),
            sum(epoch_losses['val']['cls']) / sum(epoch_losses['val']['total']),
            sum(epoch_losses['val']['obj']) / sum(epoch_losses['val']['total'])
        ]
        plt.bar([1, 2, 3], train_ratios, width, label='Train')
        plt.bar([1+width, 2+width, 3+width], val_ratios, width, label='Val')
        plt.xticks([1.1, 2.1, 3.1], ['Box', 'Cls', 'Obj'])
        plt.ylabel('Ratio to Total Loss')
        plt.title('Loss Component Ratios')
        plt.grid(True)
        plt.legend()
    else:
        # If no validation, leave these plots empty
        for i in range(4, 7):
            plt.subplot(3, 3, i)
            plt.text(0.5, 0.5, 'No validation data', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
    
    plt.suptitle(f'Comprehensive Loss Analysis (Epoch {epoch}, Batch {batch_idx})')
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(plot_path)
    plt.close()
