import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from tqdm import tqdm
import traceback

class CocoDataset(Dataset):
    def __init__(self, coco_folder, val=False,
                 augment_transform=None,
                 target_size=640, use_keypoints=False,
                 enable_mosaic=True, mosaic_prob=1.0,
                 cache_ram=True,
                 enable_mixup=True, mixup_prob=0.2, mixup_alpha=8.0):

        if not val:
            annotation_file='instances_train2017.json'
        else: # Validation settings
            annotation_file='instances_val2017.json'
            enable_mosaic = False
            enable_mixup = False

        self.coco_folder = coco_folder
        self.augment_transform = augment_transform
        self.target_size = target_size
        self.use_keypoints = use_keypoints
        self.val = val

        self.enable_mosaic = enable_mosaic if not self.val else False
        self.mosaic_prob = mosaic_prob if not self.val else 0.0
        self.enable_mixup = enable_mixup if not self.val else False
        self.mixup_prob = mixup_prob if not self.val else 0.0
        self.mixup_alpha = mixup_alpha 

        self.cache_images_in_ram = cache_ram
        self.ram_cache = None
        self._dynamically_disable_mosaic = False

        self.final_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # self.annotation_path = os.path.join(coco_folder, 'annotations', annotation_file)
        # self.image_dir = os.path.join(coco_folder, 'images',
        #                              annotation_file.replace('instances_', '').replace('.json', ''))
        
        kaggle_train_annotation = "/kaggle/input/aquarium-dataset/Aquarium Combined/train/_annotations.coco.json"
        kaggle_val_annotation = "/kaggle/input/aquarium-dataset/Aquarium Combined/valid/_annotations.coco.json"
        if not val:
            self.annotation_path = kaggle_train_annotation #os.path.join(coco_folder, 'annotations', annotation_file)
            self.image_dir = "/kaggle/input/aquarium-dataset/Aquarium Combined/train/"
        else :
            self.annotation_path = kaggle_val_annotation
            self.image_dir = "/kaggle/input/aquarium-dataset/Aquarium Combined/valid/"


        self.coco = COCO(self.annotation_path)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        self.class_map = {cat_id: i for i, cat_id in enumerate(self.coco.getCatIds())}
        self.num_classes = len(self.class_map)
        self.class_names = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        
        if not os.path.exists(self.image_dir): raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not os.path.exists(self.annotation_path): raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        if self.cache_images_in_ram:
            print(f"RAM Caching is enabled for {'train' if not val else 'val'} set.")
            self._preload_images_to_ram()

    def _apply_augment_transform(self, image_np, boxes_np, labels_np, image_identifier="image"):
        """ Helper to apply self.augment_transform and ensure target size. """
        # print(f"DEBUG: _apply_augment_transform for {image_identifier} - Before aug: boxes shape {boxes_np.shape}, labels shape {labels_np.shape}")
        if self.augment_transform:
            bboxes_for_aug = [list(b) for b in boxes_np] if boxes_np.shape[0] > 0 else []
            labels_for_aug = list(labels_np) if labels_np.shape[0] > 0 else []
            transformed = self.augment_transform(image=image_np, bboxes=bboxes_for_aug, class_labels=labels_for_aug)
            
            image_aug_np = transformed['image']
            boxes_aug_np = np.array(transformed['bboxes'], dtype=np.float32) if transformed['bboxes'] else np.zeros((0, 4), dtype=np.float32)
            labels_aug_np = np.array(transformed['class_labels'], dtype=np.int64) if transformed['class_labels'] else np.zeros((0,), dtype=np.int64)

            if boxes_aug_np.shape[0] == 0 and labels_aug_np.shape[0] != 0:
                labels_aug_np = np.array([], dtype=np.int64)
            
            # print(f"DEBUG: _apply_augment_transform for {image_identifier} - After aug: boxes shape {boxes_aug_np.shape}, labels shape {labels_aug_np.shape}")


            if image_aug_np.shape[0] != self.target_size or image_aug_np.shape[1] != self.target_size:
                rs_pad_transform = A.Compose([
                    A.LongestMaxSize(max_size=self.target_size, interpolation=cv2.INTER_LINEAR),
                    A.PadIfNeeded(min_height=self.target_size, min_width=self.target_size,
                                  border_mode=cv2.BORDER_CONSTANT, position='center', p=1.0)
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

                rs_bboxes_in = [list(b) for b in boxes_aug_np] if boxes_aug_np.shape[0] > 0 else []
                rs_labels_in = list(labels_aug_np) if labels_aug_np.shape[0] > 0 else []
                resized_data = rs_pad_transform(image=image_aug_np, bboxes=rs_bboxes_in, class_labels=rs_labels_in)
                image_aug_np = resized_data['image']
                boxes_aug_np = np.array(resized_data['bboxes'], dtype=np.float32) if resized_data['bboxes'] else np.zeros((0,4), dtype=np.float32)
                labels_aug_np = np.array(resized_data['class_labels'], dtype=np.int64) if resized_data['class_labels'] else np.zeros((0,), dtype=np.int64)
            
            return image_aug_np, boxes_aug_np, labels_aug_np
        else:
            if image_np.shape[0] != self.target_size or image_np.shape[1] != self.target_size:
                 image_np = cv2.resize(image_np, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            if not isinstance(boxes_np, np.ndarray): boxes_np = np.array(boxes_np, dtype=np.float32).reshape(-1,4)
            if not isinstance(labels_np, np.ndarray): labels_np = np.array(labels_np, dtype=np.int64).reshape(-1)
            return image_np, boxes_np, labels_np

    def set_dynamic_mosaic_disable(self, disable: bool):
        self._dynamically_disable_mosaic = disable

    def _preload_images_to_ram(self): 
        self.ram_cache = {}
        desc_text = f"Caching RAM for {'train' if not self.val else 'val'}"
        num_failed = 0
        for img_id in tqdm(self.image_ids, desc=desc_text):
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                img_path = os.path.join(self.image_dir, img_info['file_name'])
                image = cv2.imread(img_path)
                if image is None: raise IOError(f"cv2.imread returned None for: {img_path}")
                self.ram_cache[img_id] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.ram_cache[img_id] = None; num_failed += 1
        print(f"RAM Caching complete. Cached {len(self.image_ids) - num_failed}/{len(self.image_ids)} images.")
        if num_failed > 0: print(f"Warning: {num_failed} images failed to cache.")

    def __len__(self):
        return len(self.image_ids)

    def _convert_coco_to_yolo(self, bbox_coco, img_width, img_height):
        x, y, w, h = bbox_coco
        if img_width <= 0 or img_height <= 0 or w <= 0 or h <= 0: return None
        dw = 1. / img_width; dh = 1. / img_height
        cx = (x + w / 2.) * dw; cy = (y + h / 2.) * dh
        nw = w * dw; nh = h * dh
        return [cx, cy, nw, nh]

    def _load_image_and_annotations(self, idx, image_identifier="image"):
        img_id = self.image_ids[idx]
        # print(f"DEBUG: Loading {image_identifier} with ID: {img_id} (index {idx})")
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image_rgb = None

        if self.cache_images_in_ram and self.ram_cache is not None:
            cached_item = self.ram_cache.get(img_id)
            if isinstance(cached_item, np.ndarray): image_rgb = cached_item.copy()
            elif cached_item is None: return None, None, None, img_path

        if image_rgb is None:
            try:
                image_bgr = cv2.imread(img_path)
                if image_bgr is None:
                    if self.cache_images_in_ram and self.ram_cache is not None: self.ram_cache[img_id] = None
                    return None, None, None, img_path
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                if self.cache_images_in_ram and self.ram_cache is not None and self.ram_cache.get(img_id) is None : # only update if it was missing or failed
                    self.ram_cache[img_id] = image_rgb.copy()
            except Exception as e:
                if self.cache_images_in_ram and self.ram_cache is not None: self.ram_cache[img_id] = None
                return None, None, None, img_path
        
        if image_rgb is None: return None, None, None, img_path

        h0, w0 = image_rgb.shape[:2]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annotations = self.coco.loadAnns(ann_ids)
        boxes_yolo_list = []
        labels_indices_list = []
        for ann in annotations:
            bbox_coco = ann['bbox']
            if len(bbox_coco) != 4 or bbox_coco[2] <= 0 or bbox_coco[3] <= 0: continue
            cat_id = ann['category_id']
            class_idx = self.class_map.get(cat_id, -1)
            if class_idx == -1: continue
            yolo_coords = self._convert_coco_to_yolo(bbox_coco, w0, h0)
            if yolo_coords:
                boxes_yolo_list.append(yolo_coords)
                labels_indices_list.append(class_idx)
        
        boxes_np = np.array(boxes_yolo_list, dtype=np.float32) if boxes_yolo_list else np.zeros((0, 4), dtype=np.float32)
        labels_np = np.array(labels_indices_list, dtype=np.int64) if labels_indices_list else np.zeros((0,), dtype=np.int64)
        # print(f"DEBUG: Loaded {image_identifier} ID {img_id}: Raw boxes shape {boxes_np.shape}, labels shape {labels_np.shape}")
        return image_rgb, boxes_np, labels_np, img_path

    def _load_mosaic_sample(self, index): 
        s = self.target_size
        mosaic_img_canvas = np.full((s * 2, s * 2, 3), random.randint(100, 150), dtype=np.uint8)
        xc, yc = s, s 
        indices = [index] + random.sample([i for i in range(len(self)) if i != index and i < len(self.image_ids)], 3)
        final_mosaic_boxes_np_list = []; final_mosaic_labels_np_list = []
        valid_images_loaded_count = 0

        for i, idx_mosaic in enumerate(indices):
            img_part_raw, boxes_part_raw_np, labels_part_raw_np, _ = self._load_image_and_annotations(idx_mosaic, f"mosaic_part_{i}")
            if img_part_raw is None: continue
            valid_images_loaded_count +=1
            h0, w0 = img_part_raw.shape[:2]
            if i == 0: x_offset, y_offset = xc - w0, yc - h0
            elif i == 1: x_offset, y_offset = xc, yc - h0
            elif i == 2: x_offset, y_offset = xc - w0, yc
            else: x_offset, y_offset = xc, yc
            src_x1, src_y1 = max(0, -x_offset), max(0, -y_offset)
            src_x2, src_y2 = min(w0, 2*s - x_offset), min(h0, 2*s - y_offset)
            dst_x1, dst_y1 = max(0, x_offset), max(0, y_offset)
            dst_x2, dst_y2 = min(2*s, x_offset + w0), min(2*s, y_offset + h0)
            actual_w, actual_h = min(src_x2-src_x1, dst_x2-dst_x1), min(src_y2-src_y1, dst_y2-dst_y1)

            if actual_w > 0 and actual_h > 0:
                mosaic_img_canvas[dst_y1:dst_y1+actual_h, dst_x1:dst_x1+actual_w] = \
                    img_part_raw[src_y1:src_y1+actual_h, src_x1:src_x1+actual_w]
                if boxes_part_raw_np.shape[0] > 0:
                    boxes_abs_img_part = boxes_part_raw_np.copy()
                    boxes_abs_img_part[:, 0] = (boxes_part_raw_np[:, 0] - boxes_part_raw_np[:, 2]/2)*w0
                    boxes_abs_img_part[:, 1] = (boxes_part_raw_np[:, 1] - boxes_part_raw_np[:, 3]/2)*h0
                    boxes_abs_img_part[:, 2] = (boxes_part_raw_np[:, 0] + boxes_part_raw_np[:, 2]/2)*w0
                    boxes_abs_img_part[:, 3] = (boxes_part_raw_np[:, 1] + boxes_part_raw_np[:, 3]/2)*h0
                    boxes_abs_mc = boxes_abs_img_part.copy()
                    boxes_abs_mc[:, [0,2]] = boxes_abs_mc[:, [0,2]] - src_x1 + dst_x1
                    boxes_abs_mc[:, [1,3]] = boxes_abs_mc[:, [1,3]] - src_y1 + dst_y1
                    boxes_abs_mc[:,0]=np.clip(boxes_abs_mc[:,0],dst_x1,dst_x1+actual_w)
                    boxes_abs_mc[:,1]=np.clip(boxes_abs_mc[:,1],dst_y1,dst_y1+actual_h)
                    boxes_abs_mc[:,2]=np.clip(boxes_abs_mc[:,2],dst_x1,dst_x1+actual_w)
                    boxes_abs_mc[:,3]=np.clip(boxes_abs_mc[:,3],dst_y1,dst_y1+actual_h)
                    box_w_mc,box_h_mc = boxes_abs_mc[:,2]-boxes_abs_mc[:,0], boxes_abs_mc[:,3]-boxes_abs_mc[:,1]
                    valid_mask = (box_w_mc > 1) & (box_h_mc > 1)
                    if valid_mask.any():
                        vb_abs_mc, vl_mc = boxes_abs_mc[valid_mask], labels_part_raw_np[valid_mask]
                        ch, cw = mosaic_img_canvas.shape[:2]
                        cx_r,cy_r = (vb_abs_mc[:,0]+vb_abs_mc[:,2])/2/cw, (vb_abs_mc[:,1]+vb_abs_mc[:,3])/2/ch
                        w_r,h_r = box_w_mc[valid_mask]/cw, box_h_mc[valid_mask]/ch
                        final_mosaic_boxes_np_list.append(np.stack([cx_r,cy_r,w_r,h_r],axis=-1))
                        final_mosaic_labels_np_list.append(vl_mc)
        if valid_images_loaded_count < 1: raise ValueError("Mosaic: <1 valid img loaded.")
        boxes_for_aug_np=np.concatenate(final_mosaic_boxes_np_list,axis=0) if final_mosaic_boxes_np_list else np.zeros((0,4),dtype=np.float32)
        labels_for_aug_np=np.concatenate(final_mosaic_labels_np_list,axis=0) if final_mosaic_labels_np_list else np.zeros((0,),dtype=np.int64)
        return self._apply_augment_transform(mosaic_img_canvas, boxes_for_aug_np, labels_for_aug_np, "mosaic_canvas")

    def __getitem__(self, idx, force_standard=False):
        max_attempts = 5; current_attempt = 0; original_idx = idx
        applied_mixup_flag_for_debug = False 

        while current_attempt < max_attempts:
            image1_aug_np, boxes1_aug_np, labels1_aug_np = None, None, None
            
            use_mosaic_for_primary = self.enable_mosaic and not self.val and \
                                     not self._dynamically_disable_mosaic and not force_standard and \
                                     random.random() < self.mosaic_prob
            try:
                if use_mosaic_for_primary:
                    # print(f"DEBUG: Item {original_idx} - Attempting Mosaic")
                    image1_aug_np, boxes1_aug_np, labels1_aug_np = self._load_mosaic_sample(idx)
                else:
                    # print(f"DEBUG: Item {original_idx} - Attempting Single Load for img1")
                    img_raw_np, boxes_raw_np, labels_raw_np, _ = self._load_image_and_annotations(idx, "img1_raw")
                    if img_raw_np is None: raise ValueError(f"Primary image ID {self.image_ids[idx]} loading failed.")
                    image1_aug_np, boxes1_aug_np, labels1_aug_np = self._apply_augment_transform(
                        img_raw_np, boxes_raw_np, labels_raw_np, "img1_aug")

                if image1_aug_np is None: raise ValueError("Primary augmented image is None.")
                if boxes1_aug_np is None: boxes1_aug_np = np.zeros((0,4),dtype=np.float32)
                if labels1_aug_np is None: labels1_aug_np = np.zeros((0,),dtype=np.int64)

                final_image_np, final_boxes_np, final_labels_np = image1_aug_np, boxes1_aug_np, labels1_aug_np
                
                apply_this_mixup = self.enable_mixup and not self.val and \
                                   not self._dynamically_disable_mosaic and not force_standard and \
                                   not use_mosaic_for_primary and \
                                   random.random() < self.mixup_prob
                
                if apply_this_mixup:
                    # print(f"DEBUG: Item {original_idx} - Attempting Mixup")
                    idx2 = random.randint(0, len(self) - 1)
                    tries_idx2 = 0
                    while idx2 == idx and tries_idx2 < 5 and len(self) > 1:
                        idx2 = random.randint(0, len(self) - 1); tries_idx2 += 1
                    
                    if idx2 != idx or len(self) == 1:
                        img2_raw_np, boxes2_raw_np, labels2_raw_np, _ = self._load_image_and_annotations(idx2, "img2_raw_for_mixup")
                        if img2_raw_np is not None:
                            # print(f"DEBUG: Item {original_idx} - Loaded img2 for mixup (idx {idx2}). Raw boxes: {boxes2_raw_np.shape}")
                            image2_aug_np, boxes2_aug_np, labels2_aug_np = self._apply_augment_transform(
                                img2_raw_np, boxes2_raw_np, labels2_raw_np, "img2_aug_for_mixup")
                            
                            if image2_aug_np is not None:
                                # --- CORE MIXUP DEBUG ---
                                # print(f"DEBUG: Mixup - Img1 boxes: {boxes1_aug_np.shape}, Img2 boxes: {boxes2_aug_np.shape}")
                                # print(f"DEBUG: Mixup - Img1 labels: {labels1_aug_np.shape}, Img2 labels: {labels2_aug_np.shape}")
                                
                                current_mixup_alpha = self.mixup_alpha 
                                ratio = np.random.beta(current_mixup_alpha, current_mixup_alpha)
                                # print(f"DEBUG: Mixup - Alpha: {current_mixup_alpha}, Ratio: {ratio:.4f}")

                                mixed_image_np = (image1_aug_np.astype(np.float32) * ratio + \
                                                  image2_aug_np.astype(np.float32) * (1.0 - ratio)).astype(np.uint8)

                                if boxes1_aug_np.shape[0] > 0 and (boxes2_aug_np is not None and boxes2_aug_np.shape[0] > 0):
                                    mixed_boxes_np = np.concatenate((boxes1_aug_np, boxes2_aug_np), axis=0)
                                    mixed_labels_np = np.concatenate((labels1_aug_np, labels2_aug_np), axis=0)
                                elif boxes1_aug_np.shape[0] > 0:
                                    mixed_boxes_np, mixed_labels_np = boxes1_aug_np, labels1_aug_np
                                elif boxes2_aug_np is not None and boxes2_aug_np.shape[0] > 0:
                                    mixed_boxes_np, mixed_labels_np = boxes2_aug_np, labels2_aug_np
                                else: # Both sources might be empty after augmentation
                                    mixed_boxes_np = np.zeros((0, 4), dtype=np.float32)
                                    mixed_labels_np = np.zeros((0,), dtype=np.int64)
                                
                                final_image_np, final_boxes_np, final_labels_np = mixed_image_np, mixed_boxes_np, mixed_labels_np
                        #         applied_mixup_flag_for_debug = False
                        #         print(f"DEBUG: Mixup applied for item {original_idx} (img1) with item {idx2} (img2). Final boxes: {final_boxes_np.shape}")

                        #         # --- VISUAL DEBUG FOR MIXUP ---
                        #         if applied_mixup_flag_for_debug and random.random() < 0.05: # Save ~5% of mixups
                        #             try:
                        #                 os.makedirs("mixup_debug_viz", exist_ok=True)
                        #                 img1_tosave = cv2.cvtColor(image1_aug_np, cv2.COLOR_RGB2BGR)
                        #                 img2_tosave = cv2.cvtColor(image2_aug_np, cv2.COLOR_RGB2BGR)
                        #                 mix_tosave = cv2.cvtColor(final_image_np, cv2.COLOR_RGB2BGR)
                                
                        #                 # Draw boxes on img1_tosave
                        #                 for b in boxes1_aug_np: # Assuming YOLO format [cx,cy,w,h]
                        #                     x1,y1 = int((b[0]-b[2]/2)*img1_tosave.shape[1]), int((b[1]-b[3]/2)*img1_tosave.shape[0])
                        #                     x2,y2 = int((b[0]+b[2]/2)*img1_tosave.shape[1]), int((b[1]+b[3]/2)*img1_tosave.shape[0])
                        #                     cv2.rectangle(img1_tosave, (x1,y1), (x2,y2), (0,255,0), 1) # Green for img1
                                
                        #                 # Draw boxes on img2_tosave
                        #                 for b in boxes2_aug_np:
                        #                     x1,y1 = int((b[0]-b[2]/2)*img2_tosave.shape[1]), int((b[1]-b[3]/2)*img2_tosave.shape[0])
                        #                     x2,y2 = int((b[0]+b[2]/2)*img2_tosave.shape[1]), int((b[1]+b[3]/2)*img2_tosave.shape[0])
                        #                     cv2.rectangle(img2_tosave, (x1,y1), (x2,y2), (0,0,255), 1) # Red for img2
                                
                        #                 # Draw all boxes on mixed image
                        #                 for b_idx_final, b_final in enumerate(final_boxes_np):
                        #                     x1,y1 = int((b_final[0]-b_final[2]/2)*mix_tosave.shape[1]), int((b_final[1]-b_final[3]/2)*mix_tosave.shape[0])
                        #                     x2,y2 = int((b_final[0]+b_final[2]/2)*mix_tosave.shape[1]), int((b_final[1]+b_final[3]/2)*mix_tosave.shape[0])
                        #                     # Color based on origin if possible (hard without tracking, use a single color for now)
                        #                     color = (255,0,255) # Magenta for mixed
                        #                     cv2.rectangle(mix_tosave, (x1,y1), (x2,y2), color, 1)
                                
                        #                 cv2.imwrite(f"mixup_debug_viz/item{original_idx}_img1.png", img1_tosave)
                        #                 cv2.imwrite(f"mixup_debug_viz/item{idx2}_img2.png", img2_tosave)
                        #                 cv2.imwrite(f"mixup_debug_viz/item{original_idx}_mixed_with_{idx2}_ratio{ratio:.2f}.png", mix_tosave)
                        #                 print(f"SAVED Mixup debug images for item {original_idx} with {idx2}")
                        #             except Exception as e_viz:
                        #                 print(f"Error saving mixup viz: {e_viz}")
                        #         # --- END VISUAL DEBUG ---
                        #     else: print(f"DEBUG: Mixup - img2_aug_np for item {idx2} was None. Skipping mix.")
                        # else: print(f"DEBUG: Mixup - img2_raw_np for item {idx2} was None. Skipping mix.")
                else:
                    if not (self.enable_mixup and not self.val and not self._dynamically_disable_mosaic and not force_standard):
                        pass # Mixup intentionally disabled or not applicable
                    elif use_mosaic_for_primary:
                        pass # Mixup skipped because primary was mosaic
                    else: # Mixup prob not met
                        pass


                if self.final_transform:
                    if not isinstance(final_image_np, np.ndarray):
                        raise TypeError(f"Image is not NumPy array before final transform (type: {type(final_image_np)})")
                    final_bboxes_list = [list(b) for b in final_boxes_np] if final_boxes_np.shape[0] > 0 else []
                    final_labels_list = list(final_labels_np) if final_labels_np.shape[0] > 0 else []
                    transformed_final = self.final_transform(image=final_image_np, bboxes=final_bboxes_list, class_labels=final_labels_list)
                    image_tensor = transformed_final['image']
                    boxes_list_tf = transformed_final.get('bboxes', [])
                    labels_list_tf = transformed_final.get('class_labels', [])
                    boxes_tensor = torch.as_tensor(boxes_list_tf, dtype=torch.float32).reshape(-1, 4)
                    labels_tensor = torch.as_tensor(labels_list_tf, dtype=torch.long).reshape(-1)
                    if boxes_tensor.shape[0] != labels_tensor.shape[0]:
                        raise ValueError(f"Mismatch final: {boxes_tensor.shape[0]} boxes vs {labels_tensor.shape[0]} labels.")
                    return image_tensor, {'boxes': boxes_tensor, 'labels': labels_tensor}
                else: raise ValueError("self.final_transform is not defined.")
            except Exception as e:
                print(f"--- Exception in __getitem__ for original_idx {original_idx}, current_idx {idx}, attempt {current_attempt + 1} ---")
                print(f"Type: {type(e).__name__}, Message: {e}")
                traceback.print_exc(limit=2)
                # print("--------------------------------------------------------------------")
                current_attempt += 1
                force_standard = True 
                if current_attempt > 0 and len(self) > 1: idx = random.randint(0, len(self) - 1)
                if current_attempt == max_attempts:
                    # print(f"CRITICAL: Failed for original_idx {original_idx} after {max_attempts} attempts. Returning empty.")
                    empty_image_np = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
                    if self.final_transform:
                        empty_transformed = self.final_transform(image=empty_image_np, bboxes=[], class_labels=[])
                        empty_image_tensor = empty_transformed['image']
                    else: empty_image_tensor = torch.from_numpy(empty_image_np.transpose(2,0,1)).float()/255.0
                    return empty_image_tensor, {'boxes': torch.zeros((0,4)), 'labels': torch.zeros((0,),dtype=torch.long)}
