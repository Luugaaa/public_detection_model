import torch
import math
from src.config.config import CLASS_NAMES 

def collate_fn(batch, strides=[8, 16, 32], max_objects_per_pixel=1, num_classes=8, reg_max=16, center_sample_radius=1.5):
    images = [item[0] for item in batch]
    raw_targets_list = [item[1] for item in batch]

    if not images: # Handle empty batch case 
        empty_targets = {
            'target_boxes': torch.empty((0, 0, 4), dtype=torch.float32), 'target_ltrb_for_dfl': torch.empty((0, 0, 4), dtype=torch.float32),
            'target_classes': torch.empty((0, 0, num_classes), dtype=torch.float32), 'target_objectness': torch.empty((0, 0, 1), dtype=torch.float32),
            'pos_mask': torch.empty((0, 0), dtype=torch.bool), 'anchor_points_norm': torch.empty((0, 0, 2), dtype=torch.float32),
            'strides_tensor': torch.empty((0, 0, 1), dtype=torch.float32)
        }
        return torch.empty((0,3,0,0)), empty_targets, []


    batch_images = torch.stack(images, 0)
    B, C, H_img, W_img = batch_images.shape
    device = batch_images.device

    # --- Part 1: Precompute anchor/grid information ---
    all_anchor_points_norm_map_components = []
    all_strides_map_components = []
    # scale_start_indices = {}
    running_flat_idx_count = 0
    # Store grid H, W for each scale for center sampling calc
    scale_grid_dims = []


    for scale_idx, stride_val in enumerate(strides):
        H_grid = H_img // stride_val
        W_grid = W_img // stride_val
        if H_grid <= 0 or W_grid <= 0: continue
        scale_grid_dims.append({'h':H_grid, 'w':W_grid, 's':stride_val, 'start_idx': running_flat_idx_count})


        num_cells_on_scale = H_grid * W_grid
        num_anchors_on_scale = num_cells_on_scale * max_objects_per_pixel
        
        # scale_start_indices[scale_idx] = running_flat_idx_count 

        gy, gx = torch.meshgrid(torch.arange(H_grid, device=device), torch.arange(W_grid, device=device), indexing='ij')
        anchor_points_norm_scale = torch.stack([(gx.flatten() + 0.5) / W_grid, (gy.flatten() + 0.5) / H_grid], dim=-1).float()
        strides_scale = torch.full((num_cells_on_scale, 1), stride_val, device=device, dtype=torch.float32)

        all_anchor_points_norm_map_components.append(anchor_points_norm_scale.repeat_interleave(max_objects_per_pixel, dim=0))
        all_strides_map_components.append(strides_scale.repeat_interleave(max_objects_per_pixel, dim=0))
        running_flat_idx_count += num_anchors_on_scale
    
    N_total = running_flat_idx_count
    if N_total == 0: # Fallback
        empty_targets = {
            'target_boxes': torch.zeros((B, 0, 4), dtype=torch.float32, device=device), 'target_ltrb_for_dfl': torch.zeros((B, 0, 4), dtype=torch.float32, device=device),
            'target_classes': torch.zeros((B, 0, num_classes), dtype=torch.float32, device=device), 'target_objectness': torch.zeros((B, 0, 1), dtype=torch.float32, device=device),
            'pos_mask': torch.zeros((B, 0), dtype=torch.bool, device=device), 'anchor_points_norm': torch.zeros((B, 0, 2), dtype=torch.float32, device=device),
            'strides_tensor': torch.zeros((B, 0, 1), dtype=torch.float32, device=device)
        }
        return batch_images, empty_targets, raw_targets_list


    anchor_points_norm_map_template = torch.cat(all_anchor_points_norm_map_components, dim=0) # (N_total, 2)
    strides_map_template = torch.cat(all_strides_map_components, dim=0) # (N_total, 1)

    batch_target_boxes = torch.zeros((B, N_total, 4), dtype=torch.float32, device=device)
    batch_target_ltrb_for_dfl = torch.zeros((B, N_total, 4), dtype=torch.float32, device=device)
    batch_target_classes = torch.zeros((B, N_total, num_classes), dtype=torch.float32, device=device)
    batch_target_objectness = torch.zeros((B, N_total, 1), dtype=torch.float32, device=device)
    batch_pos_mask = torch.zeros((B, N_total), dtype=torch.bool, device=device)

    batch_anchor_points_norm = anchor_points_norm_map_template.unsqueeze(0).repeat(B, 1, 1)
    batch_strides_tensor = strides_map_template.unsqueeze(0).repeat(B, 1, 1)

    # --- Part 2: FCOS-Style Target Assignment ---
    for b_idx in range(B):
        target = raw_targets_list[b_idx]
        if not isinstance(target, dict) or 'boxes' not in target or 'labels' not in target: continue

        gt_boxes_img_norm_cxcywh = target['boxes'].to(device) # [cx, cy, w, h]
        gt_labels = target['labels'].to(device).long()
        num_gt = gt_boxes_img_norm_cxcywh.shape[0]
        if num_gt == 0: continue

        # Convert GT to xyxy format (normalized)
        gt_boxes_img_norm_xyxy = torch.stack([
            gt_boxes_img_norm_cxcywh[:, 0] - gt_boxes_img_norm_cxcywh[:, 2] / 2,
            gt_boxes_img_norm_cxcywh[:, 1] - gt_boxes_img_norm_cxcywh[:, 3] / 2,
            gt_boxes_img_norm_cxcywh[:, 0] + gt_boxes_img_norm_cxcywh[:, 2] / 2,
            gt_boxes_img_norm_cxcywh[:, 1] + gt_boxes_img_norm_cxcywh[:, 3] / 2,
        ], dim=-1)

        expanded_anchors_x = anchor_points_norm_map_template[:, 0].unsqueeze(1) # (N_total, 1)
        expanded_anchors_y = anchor_points_norm_map_template[:, 1].unsqueeze(1) # (N_total, 1)
        
        expanded_gt_x1 = gt_boxes_img_norm_xyxy[:, 0].unsqueeze(0) # (1, num_gt)
        expanded_gt_y1 = gt_boxes_img_norm_xyxy[:, 1].unsqueeze(0) # (1, num_gt)
        expanded_gt_x2 = gt_boxes_img_norm_xyxy[:, 2].unsqueeze(0) # (1, num_gt)
        expanded_gt_y2 = gt_boxes_img_norm_xyxy[:, 3].unsqueeze(0) # (1, num_gt)

        l_norm_matrix = expanded_anchors_x - expanded_gt_x1 # (N_total, num_gt)
        t_norm_matrix = expanded_anchors_y - expanded_gt_y1 # (N_total, num_gt)
        r_norm_matrix = expanded_gt_x2 - expanded_anchors_x # (N_total, num_gt)
        b_norm_matrix = expanded_gt_y2 - expanded_anchors_y # (N_total, num_gt)

        # Mask for anchors inside GT boxes: all l,t,r,b must be positive
        is_in_box_matrix = (l_norm_matrix > 1e-6) & \
                           (t_norm_matrix > 1e-6) & \
                           (r_norm_matrix > 1e-6) & \
                           (b_norm_matrix > 1e-6) # Shape: (N_total, num_gt)
        
        # FCOS Center Sampling (Optional but recommended)
        # For each anchor and GT pair where anchor is in GT,
        # check if anchor is also in a "center region" of the GT.
        # Center region is defined by GT center +/- (stride * center_sample_radius)
        is_in_center_region_matrix = torch.zeros_like(is_in_box_matrix, dtype=torch.bool)

        # Anchor points in pixel units (relative to grid cell structure, not image yet)
        # This needs strides_map_template
        # GT centers in normalized units (cx, cy)
        gt_cx_norm = gt_boxes_img_norm_cxcywh[:, 0].unsqueeze(0) # (1, num_gt)
        gt_cy_norm = gt_boxes_img_norm_cxcywh[:, 1].unsqueeze(0) # (1, num_gt)

        # Strides for each anchor, repeated for num_gt dimension
        s = strides_map_template # (N_total, 1)
        
        # Calculate distance from anchor to GT center in stride units
        # anchor_points_norm_map_template is (N_total, 2) [cx,cy]
        # gt_centers_norm is (1, num_gt, 2)
        # dx = |anchor_x_norm - gt_cx_norm| * W_img / stride
        # dy = |anchor_y_norm - gt_cy_norm| * H_img / stride
        dx_norm_abs = (expanded_anchors_x - gt_cx_norm).abs() # (N_total, num_gt)
        dy_norm_abs = (expanded_anchors_y - gt_cy_norm).abs() # (N_total, num_gt)

        # Convert normalized distances to center to "stride units" for comparison with radius
        # This means how many strides away is the anchor from GT center
        # (dx_norm_abs * W_img / s_val) and (dy_norm_abs * H_img / s_val)
        W_img_tensor = torch.tensor(W_img, device=device, dtype=torch.float32)
        H_img_tensor = torch.tensor(H_img, device=device, dtype=torch.float32)

        # Effective pixel distance of anchor from GT center
        dist_x_pixels = dx_norm_abs * W_img_tensor # (N_total, num_gt)
        dist_y_pixels = dy_norm_abs * H_img_tensor # (N_total, num_gt)
        
        # Max distance allowed in pixels: center_sample_radius * stride
        max_dist_pixels_x = center_sample_radius * s * W_img_tensor / W_grid # This isn't right. Radius is in strides.
        max_dist_pixels_y = center_sample_radius * s * H_img_tensor / H_grid # This isn't right.

        # Correct center sampling:
        # Anchor (grid cell center) must be within a box around GT center.
        # The box around GT center is defined by (gt_center_x +/- radius*stride, gt_center_y +/- radius*stride)
        # This means |anchor_x_pixels - gt_center_x_pixels| < radius * stride
        # |anchor_y_pixels - gt_center_y_pixels| < radius * stride
        is_in_center_region_matrix = (dist_x_pixels < center_sample_radius * s) & \
                                     (dist_y_pixels < center_sample_radius * s) # (N_total, num_gt)
        

        # Combine masks: anchor must be in GT box AND in its center region
        potential_pos_mask = is_in_box_matrix & is_in_center_region_matrix # (N_total, num_gt)

        # Resolve ambiguity: if an anchor matches multiple GTs, assign to smallest area GT
        # Calculate GT areas (normalized area w*h)
        gt_areas = gt_boxes_img_norm_cxcywh[:, 2] * gt_boxes_img_norm_cxcywh[:, 3] # (num_gt,)
        
        # For each anchor, find the GT it should be assigned to
        # if multiple GTs match this anchor (potential_pos_mask column has multiple Trues)
        gt_areas_matrix = gt_areas.unsqueeze(0).repeat(N_total, 1) # (N_total, num_gt)
        # Set area to infinity for non-matching GTs so they won't be chosen by argmin
        gt_areas_matrix[~potential_pos_mask] = float('inf')
        
        min_area_gt_indices = torch.argmin(gt_areas_matrix, dim=1) # (N_total,) -> index of GT with min area for each anchor
        
        # Create the final pos_mask for this image
        assigned_anchor_mask = (gt_areas_matrix[torch.arange(N_total), min_area_gt_indices] != float('inf'))
        batch_pos_mask[b_idx, assigned_anchor_mask] = True
        
        # Populate targets for these positive anchors
        positive_anchor_indices_img = torch.where(assigned_anchor_mask)[0] # Indices relative to N_total
        if positive_anchor_indices_img.numel() > 0:
            assigned_gt_flat_indices = min_area_gt_indices[positive_anchor_indices_img] # GT indices for each positive anchor

            # Get actual LTRB values for these assignments
            # l_norm_matrix, etc are (N_total, num_gt)
            l_assigned = l_norm_matrix[positive_anchor_indices_img, assigned_gt_flat_indices]
            t_assigned = t_norm_matrix[positive_anchor_indices_img, assigned_gt_flat_indices]
            r_assigned = r_norm_matrix[positive_anchor_indices_img, assigned_gt_flat_indices]
            b_assigned = b_norm_matrix[positive_anchor_indices_img, assigned_gt_flat_indices]
            assigned_ltrb_norm = torch.stack([l_assigned, t_assigned, r_assigned, b_assigned], dim=-1)

            # Populate objectness, classes, and box targets
            batch_target_objectness[b_idx, positive_anchor_indices_img, 0] = 1.0
            assigned_gt_labels = gt_labels[assigned_gt_flat_indices]
            epsilon_ls = 0.1 
            assigned_gt_labels_one_hot = torch.nn.functional.one_hot(assigned_gt_labels, num_classes=num_classes).float().to(device)
            smooth_targets = assigned_gt_labels_one_hot * (1.0 - epsilon_ls) + (epsilon_ls / num_classes) # Simpler version
            batch_target_classes[b_idx, positive_anchor_indices_img, :] = smooth_targets
            
            
            # For CIoU: cx,cy,w,h of the assigned GTs
            assigned_gt_boxes_cxcywh = gt_boxes_img_norm_cxcywh[assigned_gt_flat_indices]
            batch_target_boxes[b_idx, positive_anchor_indices_img, :] = assigned_gt_boxes_cxcywh

            # For DFL: l,t,r,b in STRIDE units
            # strides_for_pos_anchors = strides_map_template[positive_anchor_indices_img].squeeze(-1)
            strides_for_pos_anchors = strides_map_template[positive_anchor_indices_img] # (num_pos_anchors, 1)

            assigned_ltrb_pixel_dist = assigned_ltrb_norm * torch.tensor([W_img, H_img, W_img, H_img], device=device, dtype=torch.float32)
            assigned_ltrb_stride_units = assigned_ltrb_pixel_dist / strides_for_pos_anchors
            batch_target_ltrb_for_dfl[b_idx, positive_anchor_indices_img, :] = assigned_ltrb_stride_units.clamp(min=0, max=reg_max - 1 - 0.01)

    batch_targets_dict = {
        'target_boxes': batch_target_boxes, 'target_ltrb_for_dfl': batch_target_ltrb_for_dfl,
        'target_classes': batch_target_classes, 'target_objectness': batch_target_objectness,
        'pos_mask': batch_pos_mask, 'anchor_points_norm': batch_anchor_points_norm,
        'strides_tensor': batch_strides_tensor,
    }
    return batch_images, batch_targets_dict, raw_targets_list
