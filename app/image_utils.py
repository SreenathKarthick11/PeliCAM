# IMPORTS
import importlib.util
import numpy as np
import os
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50, vgg16, vgg19
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from skimage.segmentation import mark_boundaries
from matplotlib import cm
from matplotlib.colors import Normalize
from unittest.mock import patch
from lime import lime_image
from app.utils import resource_path
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

#---------------------------------------------------------------------------------------

class QtTqdm:
    """
    A Qt-compatible alternative to tqdm for displaying progress
    using a QProgressBar widget.

    Parameters:
        iterable (iterable): The iterable to loop through.
        progress_bar (QProgressBar): The progress bar widget to update.
    """
    def __init__(self, iterable, progress_bar):
        self.iterable = iterable
        self.progress_bar = progress_bar
        self.total = len(iterable)
        self.count = 0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.count += 1
            percent = int((self.count / self.total) * 100)
            self.progress_bar.setValue(percent)

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def close(self): pass
    def set_description(self, *args, **kwargs): pass

#----------------------------------------------------------------------------------------

    
def load_custom_model(python_file_path, class_name, weight_path):
    """
    Parameters:
        python_file_path (str): Path to the Python file containing the custom model class.
        class_name (str): Name of the model class to load from the Python file.
        weight_path (str): Path to the `.pt` or `.pth` file containing the model weights.

    Function:
        Dynamically loads a custom PyTorch model class from a given Python file, 
        instantiates the model, loads its trained weights, sets it to evaluation mode, 
        and returns the model instance.

    Returns:
        torch.nn.Module: The loaded and ready-to-use PyTorch model.
    """
    spec = importlib.util.spec_from_file_location("custom_model", python_file_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    model_class = getattr(custom_module, class_name)
    model = model_class()
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

#--------------------------------------------------------------------------------------------

def get_all_models_dict():
    """
    Function:
        Scans the models directory for all saved model weight files and returns a dictionary
        mapping model names to their full file paths.

    Returns:
        dict: A dictionary where keys are model names (filenames without extension) and values
              are the absolute paths to the model weight files.

    Notes:
        - Only files with extensions `.pth`, `.pt` are included.
        - Uses `resource_path` to support both development and PyInstaller environments.
    """
    model_dir = os.path.join("app", "models")  
    model_dir=resource_path(model_dir)
    model_dict = {}

    for filename in os.listdir(model_dir):
        if filename.endswith(".pth") or filename.endswith(".pt") or filename.endswith(".pkl"):
            model_name = os.path.splitext(filename)[0]  # Remove .pth/.pt extension
            full_path = os.path.join(model_dir, filename)
            model_dict[model_name] = full_path

    return model_dict

#------------------------------------------------------------------------------------------------
    
def get_named_conv_layers(model):
    """
    Parameters:
        model (torch.nn.Module): The PyTorch model from which to extract Conv2d layers.

    Function:
        Recursively traverses the model and collects all convolutional layers (`nn.Conv2d`)
        along with their full hierarchical names.

    Returns:
        dict: A dictionary where keys are full layer names (e.g., 'block1.conv1') and values 
              are the corresponding `nn.Conv2d` layer objects.
    """
    layers = []

    def recurse(module, prefix=""):
        for name, child in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                layers.append((layer_name, child))
            recurse(child, layer_name)

    recurse(model)
    return dict(layers)

#-------------------------------------------------------------------------------------------------
    
def get_model(name,python_file_path, class_name, weight_path):
    """
    Parameters:
        name (str): Name of the model to load (e.g., 'resnet50', 'vgg16', etc.).
        python_file_path (str): Path to the Python file containing the custom model class 
                                (used if the model is not a known base architecture).
        class_name (str): Name of the custom model class inside the Python file.
        weight_path (str): Path to the weight file for the custom model.

    Function:
        Loads a pretrained model based on its name. If the name matches a known architecture 
        (e.g., ResNet, VGG), it loads the model with the corresponding architecture and weights. 
        Otherwise, it loads a custom model using the provided Python file and class name.

    Returns:
        torch.nn.Module: A PyTorch model instance ready for inference.
    """
    # Get absolute path to the 'models' folder inside 'app'
    model_dict = get_all_models_dict()

    if name not in model_dict:
        raise ValueError(f"Model '{name}' not found in models directory.")

    model_path = model_dict[name]

    # Choose base architecture based on common model name prefixes
    if name.lower().startswith("resnet"):
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif name.lower().startswith("vgg16"):
        model = vgg16(weights=None)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif name.lower().startswith("vgg19"):
        model = vgg19(weights=None)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        model = load_custom_model(python_file_path, class_name, weight_path)

    return model

#-------------------------------------------------------------------------------------------------

def preprocess_image(img_path, size=(224, 224)):
    """
    Parameters:
        img_path (str): Path to the input image file.
        size (tuple): Desired size to resize the image (default: (224, 224)).

    Function:
        Loads an image from the given path, resizes it, normalizes it using standard
        ImageNet mean and std, and converts it into a PyTorch input tensor suitable 
        for model inference. Also returns a normalized RGB NumPy version of the image.

    Returns:
        tuple:
            - rgb_img (np.ndarray): The normalized RGB image as a NumPy array (float32, scaled 0-1).
            - input_tensor (torch.Tensor): The normalized image as a 4D PyTorch tensor ready for model input.
    """
    img = Image.open(img_path).convert("RGB").resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0
    rgb_img = img_np.copy()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    return rgb_img, input_tensor

#-------------------------------------------------------------------------------------------------

def apply_threshold_mask(cam, rgb_img, threshold):
    """
    Parameters:
        cam (np.ndarray): The class activation map (CAM), expected to be a 2D NumPy array with values in [0, 1].
        rgb_img (np.ndarray): The original RGB image as a 3D NumPy array (float32, values in [0, 1]).
        threshold (float): Threshold value to binarize the CAM (typically between 0 and 1).

    Function:
        Applies a binary threshold to the CAM and uses it as a mask over the input RGB image.
        The resulting image highlights only the regions where the CAM exceeds the threshold.

    Returns:
        np.ndarray: The RGB image with the thresholded CAM mask applied (same shape as `rgb_img`).
    """
    mask = cam.copy()
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return rgb_img * mask_3c

#------------------------------------------------------------------------------------------------------------------------------------------------

def generate_explanations(img_path, model_name, layer_name, selected_methods, threshold,python_file_path, class_name, weight_path,explain_on=True):
    """
    Parameters:
        img_path (str): Path to the input image.
        model_name (str): Name of the model to use for inference and explanation.
        layer_name (str): Name of the convolutional layer to target for CAM visualization.
        selected_methods (list): List of CAM methods to apply (e.g., ["Grad-CAM", "Layer-CAM"]).
        threshold (float): Threshold value (0–1) for masking the explanation map.
        python_file_path (str): Path to the Python file for custom model (used if not a standard model).
        class_name (str): Name of the custom model class (used if not a standard model).
        weight_path (str): Path to the weight file for the custom model.
        explain_on (bool): Whether to apply and return a masked version of the CAM (default: True).

    Function:
        Loads a model, identifies the target convolutional layer, predicts the image class, 
        and generates visual explanation maps using the selected CAM methods. Optionally, 
        applies a binary mask based on the CAM values to emphasize activated regions.

    Returns:
        dict: A dictionary containing:
              - CAM visualizations under keys like "Grad-CAM", "Grad-CAM++", etc.
              - Masked images under keys like "Grad-CAM-exp" if `explain_on` is True.

    Dependencies:
        - Uses Grad-CAM methods from `pytorch-grad-cam`.
        - Assumes `get_model`, `get_named_conv_layers`, `preprocess_image`, 
          `apply_threshold_mask`, and `show_cam_on_image` are defined elsewhere.
    """
    model = get_model(model_name,python_file_path, class_name, weight_path)
    convdict = get_named_conv_layers(model)
    target_layer = convdict[layer_name]
    rgb_img, input_tensor = preprocess_image(img_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = outputs.argmax().item()

    results = {}
    cams={}

    if "Grad-CAM" in selected_methods:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            cam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]
            cams["Grad-CAM"]=cam_map
            results["Grad-CAM"] = show_cam_on_image(rgb_img,cam_map, use_rgb=True)
            if explain_on:
                results["Grad-CAM-exp"] = apply_threshold_mask(cam_map, rgb_img, threshold)

    if "Grad-CAM++" in selected_methods:
        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as campp:
            cam_map = campp(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]
            cams["Grad-CAM++"]=cam_map
            results["Grad-CAM++"] = show_cam_on_image(rgb_img, cam_map, use_rgb=True)
            if explain_on:
                results["Grad-CAM++-exp"] = apply_threshold_mask(cam_map, rgb_img, threshold)

    if "Layer-CAM" in selected_methods:
        with LayerCAM(model=model, target_layers=[target_layer]) as lcam:
            cam_map = lcam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]
            cams["Layer-CAM"]=cam_map
            results["Layer-CAM"] = show_cam_on_image(rgb_img, cam_map, use_rgb=True)
            if explain_on:
                results["Layer-CAM-exp"] = apply_threshold_mask(cam_map, rgb_img, threshold)

    return results,cams

#-------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_lime_exp(img_path, model_name, lime_methods, num_features,progress_bar,python_file_path, class_name, weight_path):
    """
    Parameters:
        img_path (str): Relative path to the input image file.
        model_name (str): The name of the model to load for explanation.
        lime_methods (list of str): List of explanation styles: "positive", "both", "heatmap".
        num_features (int): Number of superpixels/features to display in the LIME output.
        progress_bar (QProgressBar): A Qt progress bar widget to visualize explanation progress.
        python_file_path (str): Path to custom model Python file (used if model is custom).
        class_name (str): Class name of the custom model.
        weight_path (str): Path to the model weights (for loading custom model).

    Function:
        Generates LIME (Local Interpretable Model-agnostic Explanations) visualizations 
        for the given image using the specified model. Supports various output types:
        - "positive": only positively contributing regions
        - "both": both positive and negative contributing regions
        - "heatmap": heatmap of importance values over superpixels

    Returns:
        dict: A dictionary mapping explanation types to their corresponding output images as NumPy arrays.
              Example keys: "positive", "both(+/-)", "heatmap"
    """
    # === Load and prepare image ===
    img = Image.open(resource_path(img_path)).convert("RGB").resize((224, 224))
    img_array = np.array(img)  # shape: (224, 224, 3)
    model = get_model(model_name,python_file_path, class_name, weight_path)

    # === Run LIME ===
    def predict_fn(images_np):
        images_tensor = torch.tensor(images_np.transpose((0, 3, 1, 2))).float() / 255.0
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        for i in range(images_tensor.size(0)):
            images_tensor[i] = norm(images_tensor[i])
        with torch.no_grad():
            outputs = model(images_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.numpy()

    # === Run LIME ===
    

    explainer = lime_image.LimeImageExplainer()
    with patch("lime.lime_image.tqdm", lambda x, *args, **kwargs: QtTqdm(x, progress_bar)):
        explanation = explainer.explain_instance(
            image=img_array,
            classifier_fn=predict_fn,
            top_labels=5,
            num_samples=1000,
            hide_color=0,
            batch_size=10
        )
    ind = explanation.top_labels[0]
    progress_bar.setValue(0)
    output_images = {}

    # === Masked explanations ===
    for method in lime_methods:
        if method == "positive":
            temp, mask = explanation.get_image_and_mask(ind, positive_only=True, num_features=num_features, hide_rest=True)
            boundary_img = mark_boundaries(temp, mask)
            o_img = ((1 - boundary_img) * 255).astype(np.uint8)
            white_mask = np.all(o_img == [255, 255, 255], axis=-1)
            o_img[white_mask] = [0, 0, 0]
            output_images["positive"] = o_img

        elif method == "both":
            temp, mask = explanation.get_image_and_mask(ind, positive_only=False, num_features=num_features, hide_rest=True)
            boundary_img = mark_boundaries(temp, mask)
            o_img = ((1 - boundary_img) * 255).astype(np.uint8)
            white_mask = np.all(o_img == [255, 255, 255], axis=-1)
            o_img[white_mask] = [0, 0, 0]
            output_images["both(+/-)"] = o_img

        elif method == "heatmap":
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap_2d = np.vectorize(dict_heatmap.get)(explanation.segments)

            # Normalize heatmap values to 0–1
            norm = Normalize(vmin=-np.max(np.abs(heatmap_2d)), vmax=np.max(np.abs(heatmap_2d)))
            colormap = cm.get_cmap('RdBu')  # or 'seismic', 'coolwarm', 'viridis', etc.

            # Apply colormap and convert to RGB
            heatmap_rgb = colormap(norm(heatmap_2d))[:, :, :3]  # Drop alpha channel

            output_images["heatmap"] = (heatmap_rgb * 255).astype(np.uint8)

    return output_images

#---------------------------------------------------------------------------------------------------------------------------------
def get_gt_mask_with_tolerance(mask_path, bbox_color, tolerance=40, size=(224, 224)):
    """
    Generates a binary mask from an image by selecting pixels that are approximately equal to the specified bbox color.

    Parameters:
        mask_path (str): Path to the ground truth mask image.
        bbox_color (tuple): RGB color of the bounding box to match.
        tolerance (int): Allowed deviation from bbox_color for a pixel to be included in the mask.
        size (tuple): The target size to which the mask is resized (width, height).

    Returns:
        np.ndarray: Binary mask (0 or 1), where 1 indicates pixels matching the bbox_color within tolerance.
    """
    mask_img = Image.open(mask_path).convert("RGB").resize(size)
    mask_np = np.array(mask_img).astype(np.int16)

    lower = np.array(bbox_color) - tolerance
    upper = np.array(bbox_color) + tolerance

    mask_bool = np.all((mask_np >= lower) & (mask_np <= upper), axis=-1)
    gt_mask = mask_bool.astype(np.uint8)
    return gt_mask

#---------------------------------------------------------------------------------------------------------------------------------
# ====== Saliency Score ======
def saliency_score(cam_map, threshold=0.5):
    """
    Computes the saliency score by averaging high-activation regions in a normalized CAM map.

    Parameters:
        cam_map (np.ndarray): The Class Activation Map (CAM).
        threshold (float): Threshold for selecting salient regions (default is 0.5).

    Returns:
        float: Mean intensity of the salient region; 0.0 if no salient region is found.
    """
    norm_cam = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
    salient_region = norm_cam[norm_cam > threshold]
    return salient_region.mean() if salient_region.size > 0 else 0.0

#---------------------------------------------------------------------------------------------------------------------------------
def debug_cam_and_mask(cam_bin, gt_mask, title):
    """
    Visualizes and compares the CAM binary mask and ground truth mask using matplotlib,
    and returns the visualization as an image array.

    Parameters:
        cam_bin (np.ndarray): Binary CAM image after thresholding.
        gt_mask (np.ndarray): Ground truth binary mask.
        title (str): Title for the plot.

    Returns:
        np.ndarray: RGB image array of the combined debug visualization.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(gt_mask, cmap='gray')
    axs[0].set_title("GT Mask")
    axs[1].imshow(cam_bin, cmap='gray')
    axs[1].set_title("CAM > threshold")
    axs[2].imshow(gt_mask + cam_bin, cmap='gray')
    axs[2].set_title("Overlap")
    for ax in axs:
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()

    # Convert plot to a NumPy image
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img_np = np.array(img)
    plt.close(fig)  # Prevent window from popping up
    return img_np

#---------------------------------------------------------------------------------------------------------------------------------
# ====== Overlap Metrics ======
def compute_overlap_with_mask(cam_map, mask_path,bbox_color,threshold,t):
    """
    Computes overlap metrics (IoU, recall, precision) between a CAM heatmap and a ground truth mask,
    and generates a debug visualization showing the overlap.

    Parameters:
        cam_map (np.ndarray): The CAM map (grayscale or RGB).
        mask_path (str): Path to the ground truth mask image.
        bbox_color (tuple): RGB color of the bounding box in the mask.
        threshold (float): Threshold to binarize the normalized CAM.
        t (str): Title for the debug visualization.

    Returns:
        tuple:
            - iou (float): Intersection-over-Union between CAM and mask.
            - box_recall (float): Recall = intersection / bbox area.
            - cam_precision (float): Precision = intersection / CAM area.
            - debug_image (np.ndarray): RGB visualization showing CAM, GT, and their overlap.
    """
    if cam_map.ndim == 3 and cam_map.shape[2] == 3:
        # Convert RGB to grayscale
        cam_map = np.mean(cam_map, axis=2)
    norm_cam = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
    cam_bin = (norm_cam > threshold).astype(np.uint8)
    # print(norm_cam.shape)
    # print(cam_bin.shape)
    gt_mask = get_gt_mask_with_tolerance(mask_path, bbox_color, size=cam_map.shape[:2][::-1])

    if cam_bin.shape != gt_mask.shape:
        cam_bin = resize(cam_bin, gt_mask.shape, preserve_range=True, order=0).astype(np.uint8)

    intersection = np.logical_and(cam_bin, gt_mask).sum()
    bbox_area = gt_mask.sum()
    cam_area = cam_bin.sum()
    union = np.logical_or(cam_bin, gt_mask).sum()
    debug_cam_and_mask(cam_bin,gt_mask,t)

    iou = intersection / (union + 1e-8)
    box_recall = intersection / (bbox_area + 1e-8)
    cam_precision = intersection / (cam_area + 1e-8)
    
    return iou, box_recall, cam_precision,debug_cam_and_mask(cam_bin,gt_mask,t)
#---------------------------------------------------------------------------------------------------------------------------------