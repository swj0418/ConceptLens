import os
import torch
import torchvision
import open_clip

import torch.nn.functional as F

from PIL import Image
from helpers import prepare_model, parse_generator_fp


def center_crop(tensor, target_size):
    _, _, h, w = tensor.shape  # Assuming tensor has shape [C, H, W]
    crop_h, crop_w = target_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    domain = 's2_ffhq256'
    model_path, n_layer = parse_generator_fp(domain)
    generator = prepare_model(domain=domain, model_path=model_path, device=device)

    crop = torchvision.transforms.CenterCrop(224)  # Center crop to 224x224
    toPIL = torchvision.transforms.ToPILImage()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name='ViT-H-14-quickgelu', pretrained='dfn5b')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name='ViT-H-14-quickgelu')

    # Same for ViT-H-14-quickgelu
    openai_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    openai_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)

    # Dummy direction to optimize
    text = "A person with white hair"
    ttext = tokenizer([text]).to(device)
    negative_prompts = ["A bad image", "noise", "broken image"]
    t_neg_prompts = tokenizer(negative_prompts).to(device)

    direction = torch.nn.Parameter(torch.randn(size=(1, 512)).to(device))

    # Optimizer
    optimizer = torch.optim.Adam(params=[direction], lr=0.1)

    # Create folder to save intermediate images
    output_folder = "generation_optim_test"
    os.makedirs(output_folder, exist_ok=True)
    torch.manual_seed(779)
    z = torch.randn(size=(1, 512)).to(device)

    # Generate perturbed encodings for distribution
    n_perturb = 100
    perturbations = torch.randn(size=(n_perturb, 512)).to(device) * 0.05  # Small noise
    perturbed_encodings = []

    for i, pert in enumerate(perturbations):
        perturbed_z = z + pert
        perturbed_image = generator.forward(z=perturbed_z, c=None)[0]
        perturbed_pre_img = center_crop(perturbed_image, (224, 224))
        perturbed_pre_img = (perturbed_pre_img - openai_mean) / openai_std
        perturbed_feature = model.encode_image(image=perturbed_pre_img)
        perturbed_feature = perturbed_feature / perturbed_feature.norm(dim=-1, keepdim=True)
        perturbed_encodings.append(perturbed_feature.detach().cpu())

    perturbed_encodings = torch.cat(perturbed_encodings, dim=0)  # Shape: [n_perturb, feature_dim]
    perturbed_encodings = perturbed_encodings.to(device)

    # Original encoding
    image = generator.forward(z=z, c=None)[0]
    pre_img = center_crop(image, (224, 224))
    pre_img = (pre_img - openai_mean) / openai_std
    feature = model.encode_image(image=pre_img)
    original_encoding = feature / feature.norm(dim=-1, keepdim=True)

    alpha, beta = 0.5, 0.5

    for i in range(500):
        print(f"{i}==========")
        optimizer.zero_grad()

        image = generator.forward_walking(z=z, c=None, direction=direction,
                                          layers=[_ for _ in range(13)], alpha=0.2)[0]

        # Save the generated image
        if i % 10 == 0:
            normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-5)
            pil_image = toPIL(normalized_image[0].cpu().clamp(0, 1))  # Convert tensor to PIL Image
            pil_image.save(os.path.join(output_folder, f"image_{i:03d}.png"))

        pre_img = center_crop(image, (224, 224))
        pre_img = (pre_img - openai_mean) / openai_std

        # Feature extraction
        feature = model.encode_image(image=pre_img)
        text = model.encode_text(text=ttext)
        neg_prompt_features = model.encode_text(t_neg_prompts)

        featuref = feature / feature.norm(dim=-1, keepdim=True)
        textf = text / text.norm(dim=-1, keepdim=True)
        neg_prompt_features = neg_prompt_features / neg_prompt_features.norm(dim=-1, keepdim=True)

        text_probs = featuref @ textf.T
        neg_text_probs = featuref @ neg_prompt_features.T

        # Compute "close" to perturbed distribution
        distribution_distances = 1 - (featuref @ perturbed_encodings.T)
        close = distribution_distances.mean()

        tobp = 1 - text_probs[0, 0]
        nontobp = neg_text_probs.mean()
        loss = alpha * (tobp + nontobp) + beta * close

        loss.backward()
        optimizer.step()
        print(text_probs[0], close, loss)