import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from helpers import prepare_model, parse_generator_fp

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load StyleGAN2 generator
domain = 's2_ffhq256'
model_path, n_layer = parse_generator_fp(domain)
generator = prepare_model(domain=domain, model_path=model_path, device=device)

# Load BLIP-2 model for detailed caption generation
caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
caption_model = caption_model.to(device)

def generate_image_description(image_tensor):
    """
    Generates a detailed description of the input image.

    Args:
        image_tensor (torch.Tensor): The input image tensor (C, H, W).

    Returns:
        str: The generated caption.
    """
    # Normalize and convert to PIL image
    normalized_image = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-5)
    pil_image = Image.fromarray((normalized_image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype('uint8').transpose(1, 2, 0))

    # Generate caption for the image
    inputs = caption_processor(pil_image, text="Question: Describe the image in great detail including a race, hair color, eye color, nose size, etc., Answer:", return_tensors="pt").to(device)
    caption = caption_model.generate(**inputs, max_new_tokens=256)
    caption_text = caption_processor.decode(caption[0], skip_special_tokens=True).strip()
    return caption_text


# Generate an image
torch.manual_seed(779)  # For reproducibility
z = torch.randn(size=(1, 512)).to(device)
original_image = generator.forward(z=z, c=None)[0]

# Generate and print the image description
description = generate_image_description(original_image)
print(f"Generated Caption: {description}")