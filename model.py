# model.py
import os
import torch
import torch.nn as nn
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from PIL import Image

# Monkey-patch for fastai compatibility with torchvision 0.20.1
import torchvision
if not hasattr(torchvision.models, "utils"):
    import types
    import torch.hub
    utils = types.ModuleType("utils")
    utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    torchvision.models.utils = utils

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

# Set the device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MainModel(nn.Module):
    """
    A wrapper for the generator network for image colorization.
    For inference, only the generator (net_G) is used.
    """
    def __init__(self, net_G=None, lambda_L1=100.0):
        super(MainModel, self).__init__()
        self.device = device
        self.lambda_L1 = lambda_L1
        if net_G is None:
            raise ValueError("Generator model (net_G) must be provided")
        self.net_G = net_G.to(self.device)
        self.fake_color = None

    def setup_input(self, data):
        """
        Prepares the input for the model.
        Expects a dictionary with keys 'L' (L channel) and 'ab' (dummy ab channels).
        """
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        """
        Forward pass through the generator.
        """
        self.fake_color = self.net_G(self.L)
        return self.fake_color

def build_res_unet(n_input=1, n_output=2, size=256):
    """
    Build a Dynamic U-Net based on a ResNet18 backbone.
    Args:
        n_input: Number of input channels (1 for L channel).
        n_output: Number of output channels (2 for ab channels).
        size: Spatial size of the input images.
    Returns:
        The generator network (net_G) instance.
    """
    body = create_body(resnet18(pretrained=True), n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def load_colorization_model(res_unet_weights_path="res18-unet.pt",
                            model_weights_path="final_model_weights.pt",
                            image_size=256):
    """
    Loads the colorization model along with the pre-trained weights.
    Args:
        res_unet_weights_path: Path to the generator (DynamicUnet) weights.
        model_weights_path: Path to the overall model weights.
        image_size: Input image size.
    Returns:
        An instance of MainModel in evaluation mode.
    Raises:
        FileNotFoundError: If any of the weight files are not found.
    """
    net_G = build_res_unet(n_input=1, n_output=2, size=image_size)
    if os.path.exists(res_unet_weights_path):
        net_G.load_state_dict(torch.load(res_unet_weights_path, map_location=device))
    else:
        raise FileNotFoundError(f"Generator weights not found: {res_unet_weights_path}")

    model_instance = MainModel(net_G=net_G)
    if os.path.exists(model_weights_path):
        # Load state_dict with strict=False to ignore keys not used in inference.
        model_instance.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
    else:
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    model_instance.eval()
    return model_instance

def colorize_image(model_instance, pil_image, image_size=256):
    """
    Colorizes a grayscale image using the provided model.
    Args:
        model_instance: The loaded MainModel instance.
        pil_image: A PIL Image object.
        image_size: The size to which the image is resized.
    Returns:
        A colorized image in RGB format as a NumPy array.
    """
    # Preprocess: resize and convert to tensor.
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image_rgb = pil_image.convert("RGB")
    image_tensor = preprocess(image_rgb)  # (3, H, W)

    # Convert RGB to L*a*b using skimage.
    image_lab = rgb2lab(image_tensor.permute(1, 2, 0).numpy()).astype("float32")
    image_lab_tensor = transforms.ToTensor()(image_lab)

    # Extract and normalize the L channel to [-1, 1].
    L_channel = image_lab_tensor[[0], ...] / 50.0 - 1.0
    L_channel = L_channel.unsqueeze(0).to(device)  # Add batch dimension

    # Create dummy ab channels (needed for interface consistency).
    dummy_ab = torch.zeros((1, 2, image_size, image_size), device=device)

    # Prepare input and perform the forward pass.
    model_instance.setup_input({'L': L_channel, 'ab': dummy_ab})
    with torch.no_grad():
        predicted_ab = model_instance.forward()

    # Post-process: convert predicted ab channels back to LAB then to RGB.
    predicted_ab_np = predicted_ab.squeeze(0).permute(1, 2, 0).cpu().numpy()
    L_np = L_channel.squeeze(0).permute(1, 2, 0).cpu().numpy() * 50.0 + 50.0
    lab_out = np.concatenate([L_np, predicted_ab_np * 110], axis=-1)
    rgb_out = lab2rgb(lab_out)
    return rgb_out
