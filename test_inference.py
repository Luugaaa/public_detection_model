import os
import torch
import cv2
from torch.utils.data import DataLoader
import albumentations as A

from src.dataset_handlers.full_coco_dataset_handler import CocoDataset
from src.detection_model_architecture import DetectionModel
from src.save_tools import load_model
from src.config.config import CLASS_NAMES
from src.visualization_tools import visualize_training_batch

def get_inference_transform(target_size=640):
    """
    Create a transformation pipeline for inference.
    Only includes resizing, padding and normalization without augmentations.
    """
    return A.Compose([
        # Resize/pad
        A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            position='center',
        ),
        

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.2,
        min_area=8,
    ))

def run_inference(model_path, dataset, num_images=5, confidence_threshold=0.5, output_dir='inference_results'):
    """
    Run inference on images from the dataset and visualize the results.
    
    Args:
        model_path: Path to the saved model
        dataset: Dataset to sample images from
        num_images: Number of images to process
        confidence_threshold: Confidence threshold for detections
        output_dir: Directory to save visualization results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, modeltype=DetectionModel, device=device)
    model.eval()
    
    # Create a dataloader with batch size 1 for simplicity
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            [{k: v.clone() for k, v in item[1].items()} for item in batch]
        )
    )
    
    count = 0
    processed_indices = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if count >= num_images:
                break
                
            images = images.to(device)
            visualize_training_batch(model, images, targets, CLASS_NAMES, device, 0, batch_idx, "inference_results", max_images=1)

if __name__ == "__main__":
    model_path = "train/train_190/best.pt"
    image_size = 640
    num_images_to_process = 10
    confidence_threshold = 0.4
    
    # Set up dataset
    dataset = CocoDataset(
        coco_folder='datasets/aquarium',
        augment_transform=get_inference_transform(image_size),
        val=True  # Set to False for training set
    )
    
    # Run inference
    run_inference(
        model_path=model_path,
        dataset=dataset,
        num_images=num_images_to_process,
        confidence_threshold=confidence_threshold,
        output_dir='inference_results'
    )