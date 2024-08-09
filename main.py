import torch
import torchvision
from torchvision import transforms
from typing import List, Tuple
from PIL import Image
import io

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("pretrained_vit_thulsi.pt", map_location=torch.device('cpu'))

# Updated class_names with IDs
class_names = [
    (1000, 'Chittamruthu'),
    (1001, 'Murikudi'),
    (1002, 'Neelakoduveli'),
    (1003, 'Ushamalari'),
    (1004, 'Thipali'),
    (1005, 'Uzhinja'),
    (1006, 'arootha'),
]

def pred_and_plot_image(
    image_bytes: bytes,
    model: torch.nn.Module = model,
    class_names: List[Tuple[int, str]] = class_names,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
) -> Tuple[int, str, float]:
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[Tuple[int, str]]): A list of target classes to map predictions to.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.

    Returns:
        Tuple[int, str, float]: The predicted class ID, class name, and prediction probability.
    """

    # Open image
    img = Image.open(io.BytesIO(image_bytes))

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1).item()

    # Retrieve the class ID, name, and probability
    class_id, class_name = class_names[target_image_pred_label]
    class_prob = target_image_pred_probs[0, target_image_pred_label].item()

    return class_id, class_name, class_prob

# Example usage:
# image_bytes = ...  # Your image bytes here
# class_id, class_name, class_prob = pred_and_plot_image(image_bytes)
# print(f"Predicted class ID: {class_id}, class name: {class_name}, probability: {class_prob:.4f}")
