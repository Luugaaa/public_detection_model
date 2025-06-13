import os
import torch
from tqdm import tqdm

from src.visualization_tools import find_next_train_folder, visualize_training_batch, plot_losses_from_csv
from src.config.config import CLASS_NAMES
from src.save_tools import compile_and_save_model
# from torch.profiler import profile, record_function, ProfilerActivity
import time
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.amp import autocast, GradScaler 
from collections import deque
import csv
import numpy as np 
import random
from evaluator import evaluate_model
from src.detection_model_architecture import DetectionModel


def set_optimizer_lr(optimizer, new_lr):
    """Sets the learning rate for all parameter groups in the optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


# # --- Updated train_step function with AMP enabled ---
# def train_step(images, formatted_targets, optimizer, model, loss_fn, scaler, device):
#     """ Performs a single training step using formatted targets from collate_fn """
#     images = images.to(device)
#     targets_on_device = {}
#     for key, tensor in formatted_targets.items():
#         if isinstance(tensor, torch.Tensor):
#             targets_on_device[key] = tensor.to(device)
#         else:
#             targets_on_device[key] = tensor
#     optimizer.zero_grad()
#     with autocast(enabled=scaler.is_enabled(), device_type=('cuda' if torch.cuda.is_available() else 'cpu')):
#         outputs = model(images)
#         loss_dict = loss_fn(outputs, targets_on_device)
#     total_loss = loss_dict.get('total_loss', 0)
#     box_iou_loss = loss_dict.get('box_iou_loss', torch.tensor(0.0, device=device))
#     box_dfl_loss = loss_dict.get('box_dfl_loss', torch.tensor(0.0, device=device))
#     box_loss = box_iou_loss + box_dfl_loss
#     cls_loss = loss_dict.get('cls_loss', torch.tensor(0.0, device=device))
#     obj_loss = loss_dict.get('obj_loss', torch.tensor(0.0, device=device))
#     if total_loss == 0 and (box_loss.item() != 0 or cls_loss.item() != 0 or obj_loss.item() != 0):
#          total_loss = box_loss + cls_loss + obj_loss
#     scaler.scale(total_loss).backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#     scaler.step(optimizer)
#     scaler.update()
#     return total_loss.detach(), box_loss.detach(), cls_loss.detach(), obj_loss.detach()


# # --- Updated validate function ---
# def validate(model, dataloader, loss_fn, device, epoch, class_names, save_folder):
#     model.eval()
#     total_loss_val = 0
#     total_box_loss_val = 0
#     total_cls_loss_val = 0
#     total_obj_loss_val = 0
#     batch_count = 0
#     visualized_val = False
#     os.makedirs(save_folder, exist_ok=True)
#     val_progress = tqdm(dataloader, desc=f"Validation Epoch {epoch}", leave=False)
#     with torch.no_grad():
#         for batch_idx, (images, formatted_targets, raw_targets_list) in enumerate(val_progress):
#             images = images.to(device)
#             targets_on_device = {}
#             for key, tensor in formatted_targets.items():
#                 if isinstance(tensor, torch.Tensor): targets_on_device[key] = tensor.to(device)
#                 else: targets_on_device[key] = tensor
#             outputs = model(images)
#             loss_dict = loss_fn(outputs, targets_on_device)
#             loss = loss_dict.get('total_loss', 0)
#             box_iou_loss = loss_dict.get('box_iou_loss', torch.tensor(0.0, device=device))
#             box_dfl_loss = loss_dict.get('box_dfl_loss', torch.tensor(0.0, device=device))
#             box_loss = box_iou_loss + box_dfl_loss
#             cls_loss = loss_dict.get('cls_loss', torch.tensor(0.0, device=device))
#             obj_loss = loss_dict.get('obj_loss', torch.tensor(0.0, device=device))
#             if loss == 0 and (box_loss.item() != 0 or cls_loss.item() != 0 or obj_loss.item() != 0):
#                 loss = box_loss + cls_loss + obj_loss
#             total_loss_val += loss.item()
#             total_box_loss_val += box_loss.item()
#             total_cls_loss_val += cls_loss.item()
#             total_obj_loss_val += obj_loss.item()
#             batch_count += 1
#             if not visualized_val:
#                  print(f"\nGenerating validation visualization for epoch {epoch} (first batch)...")
#                  visualize_training_batch(
#                      model=model,
#                      images=images.cpu(),
#                      targets=raw_targets_list,
#                      class_names=class_names,
#                      device=device,
#                      epoch=epoch,
#                      batch_idx=batch_idx,
#                      save_folder=save_folder,
#                  )
#                  visualized_val = True
#             if batch_idx % 10 == 0:
#                  val_progress.set_postfix({
#                      "Loss": f"{loss.item():.4f}",
#                      "Box": f"{box_loss.item():.4f}",
#                      "Cls": f"{cls_loss.item():.4f}",
#                      "Obj": f"{obj_loss.item():.4f}"
#                  })
#     avg_loss = total_loss_val / batch_count if batch_count > 0 else 0
#     avg_box_loss = total_box_loss_val / batch_count if batch_count > 0 else 0
#     avg_cls_loss = total_cls_loss_val / batch_count if batch_count > 0 else 0
#     avg_obj_loss = total_obj_loss_val / batch_count if batch_count > 0 else 0
#     print(f"Validation Epoch {epoch} Avg Loss: {avg_loss:.4f} [Box:{avg_box_loss:.4f}, Cls:{avg_cls_loss:.4f}, Obj:{avg_obj_loss:.4f}]")
#     model.train()
#     return avg_loss, avg_box_loss, avg_cls_loss, avg_obj_loss

def train_step(images, formatted_targets, optimizer, model, loss_fn, scaler, device,
               label_hints=None, pos_size_hints=None):
    """ Performs a single training step using formatted targets from collate_fn """
    images = images.to(device)
    targets_on_device = {}
    for key, tensor in formatted_targets.items():
        if isinstance(tensor, torch.Tensor):
            targets_on_device[key] = tensor.to(device)
        else:
            targets_on_device[key] = tensor

    # << NEW: Move hints to device if they are provided >>
    if label_hints:
        label_hints = [h.to(device) for h in label_hints]
    if pos_size_hints:
        pos_size_hints = [h.to(device) for h in pos_size_hints]

    optimizer.zero_grad()
    with autocast(enabled=scaler.is_enabled(), device_type=('cuda' if torch.cuda.is_available() else 'cpu')):
        # << MODIFIED: Pass hints to the model >>
        outputs = model(images, label_hints=label_hints, pos_size_hints=pos_size_hints)
        loss_dict = loss_fn(outputs, targets_on_device)

    total_loss = loss_dict.get('total_loss', 0)
    box_iou_loss = loss_dict.get('box_iou_loss', torch.tensor(0.0, device=device))
    box_dfl_loss = loss_dict.get('box_dfl_loss', torch.tensor(0.0, device=device))
    box_loss = box_iou_loss + box_dfl_loss
    cls_loss = loss_dict.get('cls_loss', torch.tensor(0.0, device=device))
    obj_loss = loss_dict.get('obj_loss', torch.tensor(0.0, device=device))
    if total_loss == 0 and (box_loss.item() != 0 or cls_loss.item() != 0 or obj_loss.item() != 0):
         total_loss = box_loss + cls_loss + obj_loss
    scaler.scale(total_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    scaler.step(optimizer)
    scaler.update()
    return total_loss.detach(), box_loss.detach(), cls_loss.detach(), obj_loss.detach()


# --- Updated validate function ---
def validate(model, dataloader, loss_fn, device, epoch, class_names, save_folder):
    model.eval()
    total_loss_val = 0
    total_box_loss_val = 0
    total_cls_loss_val = 0
    total_obj_loss_val = 0
    batch_count = 0
    visualized_val = False
    os.makedirs(save_folder, exist_ok=True)
    val_progress = tqdm(dataloader, desc=f"Validation Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch_idx, (images, formatted_targets, raw_targets_list) in enumerate(val_progress):
            images = images.to(device)
            targets_on_device = {}
            for key, tensor in formatted_targets.items():
                if isinstance(tensor, torch.Tensor): targets_on_device[key] = tensor.to(device)
                else: targets_on_device[key] = tensor

            # << MODIFIED: Pass None for hints during validation to get true performance >>
            outputs = model(images, label_hints=None, pos_size_hints=None)

            loss_dict = loss_fn(outputs, targets_on_device)
            loss = loss_dict.get('total_loss', 0)
            box_iou_loss = loss_dict.get('box_iou_loss', torch.tensor(0.0, device=device))
            box_dfl_loss = loss_dict.get('box_dfl_loss', torch.tensor(0.0, device=device))
            box_loss = box_iou_loss + box_dfl_loss
            cls_loss = loss_dict.get('cls_loss', torch.tensor(0.0, device=device))
            obj_loss = loss_dict.get('obj_loss', torch.tensor(0.0, device=device))
            if loss == 0 and (box_loss.item() != 0 or cls_loss.item() != 0 or obj_loss.item() != 0):
                loss = box_loss + cls_loss + obj_loss
            total_loss_val += loss.item()
            total_box_loss_val += box_loss.item()
            total_cls_loss_val += cls_loss.item()
            total_obj_loss_val += obj_loss.item()
            batch_count += 1
            if not visualized_val:
                 print(f"\nGenerating validation visualization for epoch {epoch} (first batch)...")
                 visualize_training_batch(
                     model=model,
                     images=images.cpu(),
                     targets=raw_targets_list,
                     class_names=class_names,
                     device=device,
                     epoch=epoch,
                     batch_idx=batch_idx,
                     save_folder=save_folder,
                 )
                 visualized_val = True
            if batch_idx % 10 == 0:
                 val_progress.set_postfix({
                     "Loss": f"{loss.item():.4f}",
                     "Box": f"{box_loss.item():.4f}",
                     "Cls": f"{cls_loss.item():.4f}",
                     "Obj": f"{obj_loss.item():.4f}"
                 })
    avg_loss = total_loss_val / batch_count if batch_count > 0 else 0
    avg_box_loss = total_box_loss_val / batch_count if batch_count > 0 else 0
    avg_cls_loss = total_cls_loss_val / batch_count if batch_count > 0 else 0
    avg_obj_loss = total_obj_loss_val / batch_count if batch_count > 0 else 0
    print(f"Validation Epoch {epoch} Avg Loss: {avg_loss:.4f} [Box:{avg_box_loss:.4f}, Cls:{avg_cls_loss:.4f}, Obj:{avg_obj_loss:.4f}]")
    model.train()
    return avg_loss, avg_box_loss, avg_cls_loss, avg_obj_loss

def evaluate(model_path, eval_val_loader):
    MODEL_PATH = model_path
    MODEL_ARCHITECTURE = DetectionModel
    IMG_SIZE = 640
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    STRIDES = [8, 16, 32]
    K = 1
    print("Setting up evaluation dataloaders...")
    if eval_val_loader:
        print("Running evaluation...")
        evaluation_results = evaluate_model(
            model_path=MODEL_PATH,
            model_type=MODEL_ARCHITECTURE,
            data_loader=eval_val_loader,
            class_names=CLASS_NAMES,
            device=DEVICE,
            conf_thres=0.001,
            iou_thres=0.7,
            img_size=IMG_SIZE,
            strides=STRIDES,
            max_objects_per_pixel=K
        )
        if evaluation_results:
            print("\nEvaluation completed successfully.")
        else:
            print("\nEvaluation failed or encountered errors.")
    else:
        print("\nSkipping evaluation due to dataloader setup error.")

# --- Updated train function ---
def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, class_names, device,
          vis_interval=200, val_epochs=10, epoch_vis_interval=10,
          warmup_epochs=3, initial_lr=3e-4, scheduler_T_max=None, scheduler_eta_min=1e-6,
          no_mosaic_epochs=200, reenable_mosaic_epoch=100, reno_mosaic_epoch=50,
          weaning_epochs=50): 

    model.to(device)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn.to(device)

    num_warmup_epochs = warmup_epochs
    target_lr = initial_lr
    warmup_batches = 0
    if num_warmup_epochs > 0:
        try:
             loader_len = len(train_dataloader)
             if loader_len > 0:
                 warmup_batches = num_warmup_epochs * loader_len
                 start_lr = 1e-7
                 print(f"Using LR warmup for {num_warmup_epochs} epochs (~{warmup_batches} batches) from {start_lr:.1e} to {target_lr:.1e}")
                 set_optimizer_lr(optimizer, start_lr)
             else:
                 print("Warning: Cannot determine train_dataloader length. Using fixed warmup steps (e.g., 1000).")
                 warmup_batches = 100
                 start_lr = 1e-7
                 print(f"Using LR warmup for {warmup_batches} batches from {start_lr:.1e} to {target_lr:.1e}")
                 set_optimizer_lr(optimizer, start_lr)
        except TypeError:
             print("Warning: Cannot determine train_dataloader length. Using fixed warmup steps (e.g., 1000).")
             warmup_batches = 100
             start_lr = 1e-7
             print(f"Using LR warmup for {warmup_batches} batches from {start_lr:.1e} to {target_lr:.1e}")
             set_optimizer_lr(optimizer, start_lr)
    else:
        set_optimizer_lr(optimizer, target_lr)
        print(f"No warmup. Initial LR set to {target_lr:.1e}")

    if scheduler_T_max is None:
        scheduler_T_max = epochs - num_warmup_epochs
    if scheduler_T_max <= 0: scheduler_T_max = epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_T_max, eta_min=scheduler_eta_min)
    print(f"Using CosineAnnealingLR scheduler with T_max={scheduler_T_max}, eta_min={scheduler_eta_min}")

    train_folder, train_run_index = find_next_train_folder()
    os.makedirs(train_folder, exist_ok=True)
    vis_folder = os.path.join(train_folder, "visualizations")
    latest_train_vis_folder = os.path.join(vis_folder, "latest_train_viz")
    latest_val_vis_folder = os.path.join(vis_folder, "latest_val_viz")
    os.makedirs(latest_train_vis_folder, exist_ok=True)
    os.makedirs(latest_val_vis_folder, exist_ok=True)
    print(f"Starting training run {train_run_index}, logs in: {train_folder}")

    csv_path = os.path.join(train_folder, "training_losses.csv")
    loss_curve_path = os.path.join(train_folder, "loss_curves.png")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'epoch', 'batch', 'type',
            'total_loss', 'box_loss', 'cls_loss', 'obj_loss', 'lr'
        ])

    p_hint = 0.0
    print(f"Progressive Learning Enabled: Hints will be phased out over {weaning_epochs} epochs.")


    log_interval = min(vis_interval, 100)
    recent_train_losses = deque(maxlen=log_interval)
    epoch_stats = { 'train': {'total': 0.0, 'box': 0.0, 'cls': 0.0, 'obj': 0.0, 'count': 0} }
    best_val_loss = float('inf')
    total_batches_trained = 0
    scaler = GradScaler()

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_stats['train'] = {'total': 0.0, 'box': 0.0, 'cls': 0.0, 'obj': 0.0, 'count': 0}


        if hasattr(train_dataloader.dataset, 'set_dynamic_mosaic_disable') and \
           hasattr(train_dataloader.dataset, 'enable_mosaic'):
            # Check if mosaic was initially enabled for this dataset instance
            dataset_mosaic_initially_on = getattr(train_dataloader.dataset, 'enable_mosaic', False)

            if dataset_mosaic_initially_on: # Only attempt to control if mosaic could have been on
                is_in_no_mosaic_phase = ((no_mosaic_epochs > 0 and epoch > epochs - no_mosaic_epochs and epoch < epochs - reenable_mosaic_epoch) or (epoch > epochs - reno_mosaic_epoch))
                
                # Get current dynamic disable state. If not present (e.g. not CocoDataset), assume False.
                # CocoDataset has _dynamically_disable_mosaic initialized to False.
                currently_dynamically_disabled = getattr(train_dataloader.dataset, '_dynamically_disable_mosaic', False)

                if is_in_no_mosaic_phase:
                    if not currently_dynamically_disabled: # If mosaic is being turned OFF for this phase
                        first_no_mosaic_epoch = max(1, epochs - no_mosaic_epochs + 1)
                        print(f"\nINFO: Disabling mosaic augmentation from epoch {first_no_mosaic_epoch} for the last {no_mosaic_epochs} scheduled epochs (current epoch: {epoch}).")
                    train_dataloader.dataset.set_dynamic_mosaic_disable(True)
                else: # Not in the "no mosaic" phase
                    if currently_dynamically_disabled: # If mosaic was dynamically disabled and now should be re-enabled
                        pass # No need to print when it's normally enabled based on initial config
                    train_dataloader.dataset.set_dynamic_mosaic_disable(False)
        
        elif no_mosaic_epochs > 0 and epoch == 1: # Print warning only once at the start if control is not possible
            print(f"\nWARNING: 'no_mosaic_epochs' ({no_mosaic_epochs}) was specified, but the train_dataloader.dataset "
                  "does not have 'set_dynamic_mosaic_disable' or 'enable_mosaic' attributes, "
                  "or mosaic was not initially enabled on the dataset. "
                  "The 'no_mosaic_epochs' setting may not have the intended effect.")
        # --- End of Mosaic disabling logic ---


        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for batch_idx, (images, formatted_targets, raw_targets_list) in enumerate(train_progress):
            current_global_batch = total_batches_trained
            current_lr = optimizer.param_groups[0]['lr']

            if warmup_batches > 0 and total_batches_trained < warmup_batches:
                 lr_scale = min(1.0, float(current_global_batch + 1) / warmup_batches)
                 new_lr = start_lr + lr_scale * (target_lr - start_lr)
                 set_optimizer_lr(optimizer, new_lr)
                 current_lr = new_lr
            elif warmup_batches > 0 and total_batches_trained == warmup_batches:
                 print(f"\nWarmup complete at batch {total_batches_trained}. LR set to {target_lr:.1e}")
                 set_optimizer_lr(optimizer, target_lr)
                 current_lr = target_lr

            label_hints_to_pass = None
            pos_size_hints_to_pass = None

            # Decide whether to use hints for this iteration based on p_hint
            # random.random() > p_hint means we USE the hint
            if random.random() > p_hint:
                # Extract hints from the raw targets list.
                # Your raw_targets_list is perfect for this.
                label_hints_to_pass = [t['labels'] for t in raw_targets_list]
                # Assuming your boxes are already in [cx, cy, w, h] format
                pos_size_hints_to_pass = [t['boxes'] for t in raw_targets_list]

            loss, box_loss, cls_loss, obj_loss = train_step(
                images, formatted_targets, optimizer, model, loss_fn, scaler, device,
                label_hints=label_hints_to_pass,
                pos_size_hints=pos_size_hints_to_pass
            )
            total_batches_trained += 1
            # loss, box_loss, cls_loss, obj_loss = train_step(
            #     images, formatted_targets, optimizer, model, loss_fn, scaler, device
            # )
            # total_batches_trained += 1
            
            loss_item = loss.item(); box_loss_item = box_loss.item(); cls_loss_item = cls_loss.item(); obj_loss_item = obj_loss.item()
            recent_train_losses.append(loss_item)
            epoch_stats['train']['total'] += loss_item; epoch_stats['train']['box'] += box_loss_item
            epoch_stats['train']['cls'] += cls_loss_item; epoch_stats['train']['obj'] += obj_loss_item
            epoch_stats['train']['count'] += 1

            if batch_idx % log_interval == 0:
                 avg_recent_total = np.mean(recent_train_losses) if recent_train_losses else 0
                 train_progress.set_postfix({
                     "Avg Loss": f"{avg_recent_total:.4f}",
                     "Box": f"{box_loss_item:.4f}", "Cls": f"{cls_loss_item:.4f}", "Obj": f"{obj_loss_item:.4f}",
                     "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
                 })

            is_last_batch = (batch_idx == len(train_dataloader) - 1) if hasattr(train_dataloader, '__len__') else False
            if is_last_batch and (epoch % epoch_vis_interval == 0 or epoch==1 or epoch==3):
                 with open(csv_path, 'a', newline='') as csvfile:
                     writer = csv.writer(csvfile)
                     writer.writerow([
                         epoch, batch_idx, 'train_batch',
                         loss_item, box_loss_item, cls_loss_item, obj_loss_item,
                         optimizer.param_groups[0]['lr']
                     ])
                 plot_losses_from_csv(
                     csv_path,
                     output_folder=loss_curve_path,
                     smoothing_window=min(50, vis_interval // 2)
                 )
                 print(f"\nGenerating training visualization for epoch {epoch}, batch {batch_idx}...")
                 visualize_training_batch(
                     model=model, images=images.cpu(), targets=raw_targets_list,
                     class_names=class_names, device=device, epoch=epoch,
                     batch_idx=batch_idx, save_folder=latest_train_vis_folder
                 )

        if epoch < weaning_epochs:
            p_hint = (epoch + 1) / weaning_epochs
        else:
            # After weaning_epochs, hints are permanently disabled
            p_hint = 1.0
            
        if total_batches_trained > warmup_batches:    
            scheduler.step()

        if epoch_stats['train']['count'] > 0:
            avg_epoch_train_loss = epoch_stats['train']['total'] / epoch_stats['train']['count']
            avg_epoch_train_box = epoch_stats['train']['box'] / epoch_stats['train']['count']
            avg_epoch_train_cls = epoch_stats['train']['cls'] / epoch_stats['train']['count']
            avg_epoch_train_obj = epoch_stats['train']['obj'] / epoch_stats['train']['count']
            print(f"Epoch {epoch} Average Train Loss: {avg_epoch_train_loss:.4f} [Box:{avg_epoch_train_box:.4f}, Cls:{avg_epoch_train_cls:.4f}, Obj:{avg_epoch_train_obj:.4f}]")
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    epoch, 'EPOCH_END', 'train_epoch',
                    avg_epoch_train_loss, avg_epoch_train_box, avg_epoch_train_cls, avg_epoch_train_obj,
                    optimizer.param_groups[0]['lr']
                ])

        if val_dataloader and (epoch % val_epochs == 0 or epoch == epochs or epoch==1):
            print(f"\n--- Starting Validation run for Epoch {epoch} ---")
            avg_val_loss, avg_val_box, avg_val_cls, avg_val_obj = validate(
                model, val_dataloader, loss_fn, device, epoch, class_names,
                save_folder=latest_val_vis_folder
            )
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    epoch, 'EPOCH_END', 'val_epoch',
                    avg_val_loss, avg_val_box, avg_val_cls, avg_val_obj,
                    optimizer.param_groups[0]['lr']
                ])
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(train_folder, "best.pt")
                save_content = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(save_content, best_path)
                print(f"*** New best validation loss {best_val_loss:.4f}. Saved best model to {best_path} ***")
            print(f"--- Finished Validation run for Epoch {epoch} ---")
            model.train()

        if epoch==1 or epoch%50==0 :
            # Ensure best_path is defined (e.g., from a save, or fallback to last.pt if best.pt not saved yet)
            current_best_path = os.path.join(train_folder, "best.pt")
            if not os.path.exists(current_best_path):
                 current_best_path = os.path.join(train_folder, "last.pt") # Fallback
            if os.path.exists(current_best_path):
                 evaluate(current_best_path, val_dataloader)
            else:
                 print(f"Skipping evaluation at epoch {epoch}, model checkpoint not found.")


        last_path = os.path.join(train_folder, "last.pt")
        save_content = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'last_val_loss': avg_val_loss if 'avg_val_loss' in locals() and val_dataloader and (epoch % val_epochs == 0 or epoch == epochs or epoch==1) else float('inf'),
            'best_val_loss': best_val_loss,
        }
        torch.save(save_content, last_path)

    print("Training complete.")
    try:
        plot_losses_from_csv(
            csv_path,
            output_folder=loss_curve_path,
            smoothing_window=50
        )
        print(f"Final loss curves saved to {loss_curve_path}")
    except Exception as e:
        print(f"Failed to generate final loss plot: {e}")
