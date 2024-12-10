import os
import torch
import torchvision
import open_clip
import torch.nn.functional as F

from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from helpers import prepare_model, parse_generator_fp

def center_crop(tensor, target_size):
    _, _, h, w = tensor.shape  # Assuming tensor has shape [C, H, W]
    crop_h, crop_w = target_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

if __name__ == '__main__':
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load StyleGAN2 generator
    domain = 's2_ffhq256'
    model_path, n_layer = parse_generator_fp(domain)
    generator = prepare_model(domain=domain, model_path=model_path, device=device)

    # Transforms and model setup
    crop = torchvision.transforms.CenterCrop(224)  # Center crop to 224x224
    toPIL = torchvision.transforms.ToPILImage()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name='ViT-H-14-quickgelu', pretrained='dfn5b')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name='ViT-H-14-quickgelu')

    # Load BLIP-2 model for detailed caption generation
    caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    caption_model = caption_model.to(device)

    # Normalization constants for OpenAI's CLIP
    openai_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    openai_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)

    # Positive and negative prompts
    positive_text = "A man with white hair"
    ttext = tokenizer([positive_text]).to(device)
    negative_prompts = ["A bad image", "noise"]
    t_neg_prompts = tokenizer(negative_prompts).to(device)

    # Encode text embeddings
    text_embedding = model.encode_text(ttext)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    neg_prompt_features = model.encode_text(t_neg_prompts)
    neg_prompt_features = neg_prompt_features / neg_prompt_features.norm(dim=-1, keepdim=True)

    # Dummy direction to optimize
    direction = torch.nn.Parameter(torch.randn(size=(1, 512)).to(device))

    # Optimizer
    optimizer = torch.optim.Adam(params=[direction], lr=0.1)

    # Create folder to save intermediate images
    output_folder = "generation_optim_test"
    os.makedirs(output_folder, exist_ok=True)
    torch.manual_seed(779)
    z = torch.randn(size=(1, 512)).to(device)

    # Generate the original image and its caption
    original_image = generator.forward(z=z, c=None)[0]
    normalized_image = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-5)
    pil_image = toPIL(normalized_image[0].cpu().clamp(0, 1))

    # Generate caption for the original image
    inputs = caption_processor(images=pil_image, text="Describe this image in detail.", return_tensors="pt").to(device)
    caption = caption_model.generate(**inputs, max_new_tokens=128)
    caption_text = caption_processor.decode(caption[0], skip_special_tokens=True)
    print(f"Generated Caption: {caption_text}")
    caption_model = None

    # Encode the caption as a positive prompt
    caption_encoded = tokenizer([caption_text]).to(device)
    caption_feature = model.encode_text(caption_encoded)
    caption_feature = caption_feature / caption_feature.norm(dim=-1, keepdim=True)

    # Perturbed encodings
    n_perturb = 100
    perturbations = torch.randn(size=(n_perturb, 512)).to(device) * 0.05  # Small noise
    perturbed_encodings = []

    for pert in perturbations:
        perturbed_z = z + pert
        perturbed_image = generator.forward(z=perturbed_z, c=None)[0]
        perturbed_pre_img = center_crop(perturbed_image, (224, 224))
        perturbed_pre_img = (perturbed_pre_img - openai_mean) / openai_std
        perturbed_feature = model.encode_image(image=perturbed_pre_img)
        perturbed_feature = perturbed_feature / perturbed_feature.norm(dim=-1, keepdim=True)
        perturbed_encodings.append(perturbed_feature.detach().cpu())

    perturbed_encodings = torch.cat(perturbed_encodings, dim=0).to(device)  # Shape: [n_perturb, feature_dim]

    alpha, beta = 0.5, 0.5  # Weights for loss terms

    for i in range(500):
        print(f"Step {i}==========")
        optimizer.zero_grad()

        # Generate an image
        image = generator.forward_walking(z=z, c=None, direction=direction,
                                          layers=[_ for _ in range(13)], alpha=0.2)[0]

        # Save the generated image
        if i % 10 == 0:
            normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-5)
            pil_image = toPIL(normalized_image[0].cpu().clamp(0, 1))
            pil_image.save(os.path.join(output_folder, f"image_{i:03d}.png"))

        # Feature extraction
        pre_img = center_crop(image, (224, 224))
        pre_img = (pre_img - openai_mean) / openai_std
        feature = model.encode_image(image=pre_img)
        feature = feature / feature.norm(dim=-1, keepdim=True)

        # Loss calculations
        pos_similarity = feature @ text_embedding.T
        pos_caption_similarity = feature @ caption_feature.T
        neg_similarity = feature @ neg_prompt_features.T

        text_loss = 1 - pos_similarity[0, 0]
        caption_loss = 1 - pos_caption_similarity[0, 0]
        neg_loss = neg_similarity.mean()

        # loss = alpha * (text_loss + caption_loss) + beta * neg_loss
        loss = alpha * (text_loss) + beta * neg_loss

        loss.backward(retain_graph=True)
        optimizer.step()

        print(f"Text Sim: {pos_similarity[0, 0]:.4f}, Caption Sim: {pos_caption_similarity[0, 0]:.4f}, Neg Sim: {neg_loss:.4f}, Loss: {loss:.4f}")