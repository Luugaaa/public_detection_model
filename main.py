import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from xml.etree import ElementTree as ET
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.dataset_handlers.full_coco_dataset_handler import CocoDataset

from src.config.config import CLASS_NAMES
from src.training import train
from src.save_tools import load_model
from src.detection_model_architecture import DetectionModel
from src.batchloss import DFL_CIoU_Loss
from src.collate_fn import collate_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device :", device)


def get_train_transform(target_size=640):
    """
    Returns an Albumentations Compose pipeline with enhanced augmentations for training.
    NOTE: Does not include Mosaic/MixUp which require Dataset/Collate modifications.
    """
    return A.Compose([
        # --- Geometric Augmentations ---
        A.RandomSizedBBoxSafeCrop(height=target_size, width=target_size, erosion_rate=0.2, p=0.3), 

        # Basic Flips 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), # Vertical flips less common unless data symmetry allows

        A.ShiftScaleRotate(
            shift_limit=0.1,    
            scale_limit=0.2,    
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.8 
        ),

        # --- Color Augmentations ---
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
        A.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.7, hue=0.015, p=0.8), # High probability
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), 

        # --- Occlusion / Dropout ---
        A.CoarseDropout(
            p=0.4
        ),

        # --- Resize and Pad (Important to do AFTER geometric transforms that change size/shape) ---
        A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            position='center',
            p=1.0 # Always apply padding to ensure target size
        ),

    ], bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_visibility=0.2, 
        min_area=16,
    ))


def get_val_transform(target_size=640):
    return A.Compose([
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            position='center',
            p=1.0
        ),
        A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LINEAR),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225]),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Initialize components
# model = DetectionModel(num_classes=len(CLASS_NAMES)).to(device)
# model = load_model("train/train_128/best.pt", modeltype=DetectionModel, device=device).to(device)
model = load_model("/kaggle/input/mid-updated-model-140625/best-5.pt", modeltype=DetectionModel, device=device).to(device)


LEARNING_RATE=1e-4
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001) #, weight_decay=0.0005)
loss_fn = DFL_CIoU_Loss(num_classes=len(CLASS_NAMES))

BATCH_SIZE=10
IM_SIZE=640

train_dataset = CocoDataset(
    # coco_folder='datasets/aquarium',
    coco_folder='/content/drive/MyDrive/Aquarium Combined',
    augment_transform=get_train_transform(IM_SIZE)
)


val_dataset = CocoDataset(
    # coco_folder='datasets/aquarium',
    coco_folder='/content/drive/MyDrive/Aquarium Combined',
    augment_transform=get_val_transform(IM_SIZE),
    val=True
)



def denormalize_image(tensor_image):
    """Denormalizes a tensor image from (0,1) range and typical normalization."""
    # for normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if tensor_image.device != mean.device: # Ensure devices match
        mean = mean.to(tensor_image.device)
        std = std.to(tensor_image.device)

    img = tensor_image.cpu().clone() * std + mean
    img = img.clamp(0, 1)  # Clamp to [0,1] after denormalization
    return img.permute(1, 2, 0).numpy()


def visualize_collate_fn_output(dataloader, num_batches_to_viz=1, class_names=None, strides=[8,16,32], reg_max=16, is_dfl_targets_present=True):
    """
    Visualizes the output of the collate_fn by drawing assigned ground truth boxes.
    Args:
        dataloader: The PyTorch DataLoader using the custom collate_fn.
        num_batches_to_viz: How many batches to visualize.
        class_names: List of class names for labels.
        strides: Model strides (for context if needed, not directly used for drawing assigned GT here).
        reg_max: DFL reg_max (for context if needed).
        is_dfl_targets_present: Boolean, indicates if DFL-specific targets like 'target_ltrb_for_dfl'
                               are generated by the collate_fn. This function primarily visualizes
                               the 'target_boxes' which should always contain cx,cy,w,h of the assigned GT.
    """
    print("Visualizing collate_fn output (assigned ground truths)...")
    for batch_idx, (batch_images, batch_targets_dict, raw_targets_list) in enumerate(dataloader):
        if batch_idx >= num_batches_to_viz:
            break

        print(f"\n--- Batch {batch_idx + 1} ---")
        B, C, H_img_tensor, W_img_tensor = batch_images.shape
        device = batch_images.device

        # Unpack targets needed for visualization
        target_boxes_gt = batch_targets_dict.get('target_boxes') # (B, N_total, 4)
        target_classes_gt = batch_targets_dict.get('target_classes') # (B, N_total, num_classes)
        pos_mask_gt = batch_targets_dict.get('pos_mask')           # (B, N_total)

        if target_boxes_gt is None or target_classes_gt is None or pos_mask_gt is None:
            print("  Skipping batch visualization: 'target_boxes', 'target_classes', or 'pos_mask' missing from collate_fn output.")
            continue

        for b_img_idx in range(B):
            print(f"Image {b_img_idx + 1}/{B} in batch:")
            img_tensor = batch_images[b_img_idx]
            try:
                img_to_show = denormalize_image(img_tensor)
            except NameError: 
                print("Warning: denormalize_image function not found. Displaying raw tensor permuted.")
                img_to_show = img_tensor.cpu().permute(1, 2, 0).numpy()
                if img_to_show.min() < 0 or img_to_show.max() > 1 and img_to_show.max() <= 255 : 
                     img_to_show = np.clip(img_to_show, 0, 1) if img_to_show.max() <=1 else np.clip(img_to_show,0,255).astype(np.uint8)


            fig, ax = plt.subplots(1, figsize=(10, 10)) 
            ax.imshow(img_to_show)
            title = f"Batch {batch_idx+1}, Image {b_img_idx+1} - Assigned GTs (Red) & Raw GTs (Green--)"
            if is_dfl_targets_present:
                title += " (DFL Targets also present in batch)"
            ax.set_title(title)

            num_pos_assigned_in_image = 0
            active_pos_indices = pos_mask_gt[b_img_idx].nonzero(as_tuple=False).squeeze(-1)

            for flat_idx in active_pos_indices:
                num_pos_assigned_in_image +=1
                # Get the assigned GT box [cx, cy, w, h] - normalized
                box_cxcywh = target_boxes_gt[b_img_idx, flat_idx, :4] # Ensure it's :4 if reg_max related stuff is there

                cx, cy, w_box, h_box = box_cxcywh.tolist() # Renamed w,h to avoid conflict

                # Convert YOLO [cx, cy, w, h] to [x_min, y_min, width, height] for matplotlib
                x_min = (cx - w_box / 2) * W_img_tensor
                y_min = (cy - h_box / 2) * H_img_tensor
                box_width_px = w_box * W_img_tensor
                box_height_px = h_box * H_img_tensor

                # Get class label
                class_probs = target_classes_gt[b_img_idx, flat_idx]
                class_idx = class_probs.argmax().item() # one-hot or smoothed one-hot
                label_name = class_names[class_idx] if class_names and 0 <= class_idx < len(class_names) else f"Cls:{class_idx}"

                rect = patches.Rectangle((x_min, y_min), box_width_px, box_height_px,
                                         linewidth=1.5, edgecolor='r', facecolor='none') # Made thicker
                ax.add_patch(rect)
                ax.text(x_min, y_min - 10, f"{label_name}", color='red', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, pad=0))


            print(f"  Total positive assignments (assigned GTs) in this image: {num_pos_assigned_in_image}")
            if num_pos_assigned_in_image == 0:
                print("  No GT boxes assigned for this image in the collated targets (or no GTs in raw).")

            # Also draw raw GT boxes for comparison
            if raw_targets_list and b_img_idx < len(raw_targets_list):
                raw_target_for_img = raw_targets_list[b_img_idx]
                if isinstance(raw_target_for_img, dict) and 'boxes' in raw_target_for_img and raw_target_for_img['boxes'].numel() > 0:
                    for gt_raw_idx in range(raw_target_for_img['boxes'].shape[0]):
                        raw_box_cxcywh = raw_target_for_img['boxes'][gt_raw_idx]
                        raw_label_idx = raw_target_for_img['labels'][gt_raw_idx].item()
                        raw_label_name = class_names[raw_label_idx] if class_names and 0 <= raw_label_idx < len(class_names) else f"RawCls:{raw_label_idx}"

                        rcx, rcy, rw, rh = raw_box_cxcywh.tolist()
                        rx_min = (rcx - rw / 2) * W_img_tensor
                        ry_min = (rcy - rh / 2) * H_img_tensor
                        rbox_width_px = rw * W_img_tensor
                        rbox_height_px = rh * H_img_tensor

                        raw_rect = patches.Rectangle((rx_min, ry_min), rbox_width_px, rbox_height_px,
                                                 linewidth=1, edgecolor='g', linestyle='--', facecolor='none')
                        ax.add_patch(raw_rect)
                        # ax.text(rx_min + 5, ry_min + 5, f"{raw_label_name}", color='green', fontsize=7) 
                else:
                    print("  No raw GT boxes (or raw_targets_list format issue) for this image.")
            else:
                print("  Raw targets list not available for this image index.")


            os.makedirs("collate_vis", exist_ok=True)
            filename = os.path.join("collate_vis", f"batch{batch_idx}_img{b_img_idx}.png") 
            try:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
                print(f"  Saved visualization to {filename}")
            except Exception as e:
                print(f"  Error saving plot {filename}: {e}")
            plt.close(fig)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=3,
    # persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=3, 
    # persistent_workers=True
)
# Training
epochs=300
train(model, train_loader, val_dataloader=val_loader, optimizer=optimizer, loss_fn=loss_fn, 
        class_names=CLASS_NAMES, device=device, epochs=epochs,
        initial_lr=LEARNING_RATE)
