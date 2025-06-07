import torch
import cv2
import numpy as np
from pathlib import Path
from src.detection_model_architecture import DetectionModel

def compile_and_save_model(model, save_path, example_input=None, compile_mode='reduce-overhead'):
    """
    Compile a PyTorch model using TorchScript and save it
    
    Args:
        model (nn.Module): The model to compile and save
        save_path (str): Path to save the compiled model
        example_input (torch.Tensor, optional): Example input for tracing
        compile_mode (str): PyTorch compilation mode
    """
    model.eval()
    
    num_classes = model.head_p3.num_classes
    
    # 1. Save standard PyTorch model (safest option)
    standard_path = Path(save_path).with_suffix('.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'CustomDetector',
        'num_classes': num_classes
    }, standard_path)
    print(f"Standard model saved to {standard_path}")
    

def load_model(model_path, modeltype, device='cpu', compile_mdl=False, compile_mode='reduce-overhead'):
    """
    Load a saved model checkpoint, handling different saving formats and inferring parameters if needed.

    Args:
        model_path (str): Path to the saved model (.pt, .pth, .torchscript.pt, .script.pt).
        modeltype (Type[nn.Module]): The class of the model to instantiate (e.g., YOLOLikeDetector).
        device (str): Device to load the model to ('cpu', 'cuda:0', etc.).
        compile_mdl (bool): Whether to compile the model using torch.compile after loading.
        compile_mode (str): PyTorch compilation mode if compile_mdl is True.

    Returns:
        nn.Module: The loaded and initialized model.
    """
    path = Path(model_path)
    print(f"Loading model from: {model_path}")
    device = torch.device(device) # Ensure device is a torch.device object

    if path.suffix in ['.torchscript.pt', '.script.pt']:
        # Load TorchScript model
        try:
            model = torch.jit.load(model_path, map_location=device)
            print(f"Successfully loaded TorchScript model from {model_path}")
        except Exception as e:
            print(f"Error loading TorchScript model: {e}")
            raise
    elif path.suffix in ['.pt', '.pth']:
        # Load standard PyTorch checkpoint dictionary
        try:
            checkpoint = torch.load(model_path, map_location=device)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            raise

        # --- Determine Model Parameters ---
        saved_num_classes = checkpoint.get('num_classes', None)
        saved_k = checkpoint.get('max_objects_per_pixel', None)

        if saved_num_classes is not None:
            num_classes = saved_num_classes
            print(f"Using num_classes={num_classes} from checkpoint.")
        else:
            # Try to infer num_classes if K is known
            print("num_classes not found in checkpoint, attempting inference...")
            if saved_k is None:
                 # Cannot infer num_classes without K, assume K=1 if necessary
                 print("Warning: 'max_objects_per_pixel' (K) not found in checkpoint. Assuming K=1 for num_classes inference.")
                 saved_k = 1 # Default assumption, might be wrong!

            cls_head_weight_key = 'heads.0.cls_pred.weight' # Correct key for the first head's cls layer
            if 'model_state_dict' in checkpoint and cls_head_weight_key in checkpoint['model_state_dict']:
                cls_head_weight = checkpoint['model_state_dict'][cls_head_weight_key]
                out_channels = cls_head_weight.shape[0] # Shape: [out_C, in_C, kH, kW]

                if out_channels % saved_k != 0:
                    print(f"Error: Cannot cleanly infer num_classes. Final layer output channels ({out_channels}) not divisible by assumed K ({saved_k}).")
                    raise ValueError("Failed to infer num_classes from checkpoint.")

                num_classes = out_channels // saved_k
                print(f"Inferred num_classes={num_classes} from checkpoint using K={saved_k}.")
            else:
                print(f"Error: Cannot infer num_classes. Key '{cls_head_weight_key}' not found in checkpoint state_dict.")
                raise ValueError(f"Could not determine num_classes. Please ensure it's saved in the checkpoint or provide '{cls_head_weight_key}'.")

        if saved_k is None:
            # If K wasn't saved and wasn't needed for inference (because num_classes was saved)
            # we still need it for model instantiation. Assume 1.
            print("Warning: 'max_objects_per_pixel' (K) not found in checkpoint. Assuming K=1 for model instantiation.")
            max_objects_per_pixel = 1
        else:
            max_objects_per_pixel = saved_k
            print(f"Using max_objects_per_pixel={max_objects_per_pixel} from checkpoint (or assumption).")


        # --- Instantiate Model ---
        try:
            # Pass all necessary arguments known at load time
            # Add other required args like fpn_feat_channels if they aren't the default
            model = modeltype(
                num_classes=num_classes,
                max_objects_per_pixel=max_objects_per_pixel
                # Add other args if needed, e.g., fpn_feat_channels=checkpoint.get('fpn_channels', 256)
            )
            print(f"Instantiated model {modeltype.__name__} with {num_classes} classes and K={max_objects_per_pixel}.")
        except Exception as e:
            print(f"Error instantiating model {modeltype.__name__}: {e}")
            raise

        # --- Load State Dict ---
        if 'model_state_dict' in checkpoint:
            try:
                # Load the state dict. Set strict=False if you know some layers might mismatch
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print("Successfully loaded model_state_dict.")
            except RuntimeError as e:
                 print(f"Error loading state_dict (strict=True): {e}")
                 print("Attempting load with strict=False...")
                 try:
                     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                     print("Successfully loaded model_state_dict with strict=False (some keys might have mismatched).")
                 except Exception as e2:
                     print(f"Error loading state_dict even with strict=False: {e2}")
                     raise e2 # Reraise the error if loading fails completely
            except Exception as e:
                 print(f"An unexpected error occurred during state_dict loading: {e}")
                 raise
        else:
            print("Error: 'model_state_dict' not found in checkpoint.")
            raise KeyError("'model_state_dict' missing from the loaded checkpoint file.")

        model.to(device) # Move model to the specified device

        # --- Compile Model (Optional) ---
        if compile_mdl:
            if hasattr(torch, 'compile'):
                try:
                    print(f"Attempting to compile model with mode: {compile_mode}...")
                    model = torch.compile(model, mode=compile_mode)
                    print("Model compiled successfully.")
                except Exception as e:
                    print(f"Warning: Model compilation failed: {e}. Proceeding with uncompiled model.")
            else:
                print("Warning: torch.compile not available in this PyTorch version. Model not compiled.")

    else:
        print(f"Error: Unsupported model file extension '{path.suffix}'. Please use .pt, .pth, or .torchscript.pt")
        raise ValueError("Unsupported model file type")

    # Set model to evaluation mode after loading
    if hasattr(model, 'eval'):
        model.eval()
        print("Model set to evaluation mode.")

    return model