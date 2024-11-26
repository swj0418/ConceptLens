import torch
from diffusers import DDIMPipeline, schedulers
from PIL import Image
import numpy as np

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


model_id = "google/ddpm-celebahq-256"

# load model and scheduler
# pipe = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe = DDIMPipeline.from_pretrained(model_id)

# Replace the scheduler with a new one that has the desired number of steps
pipe.scheduler = schedulers.DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(num_inference_steps=10)  # Set inference steps


# # run pipeline in inference (sample random noise and denoise)
# image = pipe(num_inference_steps=10)[0][0]
# # save image
# image.save("ddim_generated_image.png")

# Define reverse schedule (simplified, requires proper alpha/beta schedules)
# Example assumes pipe.scheduler has `alphas_cumprod` defined
scheduler = pipe.scheduler
timesteps = scheduler.timesteps
alphas_cumprod = scheduler.alphas_cumprod.to(pipe.device)
print("Timesteps: ", len(timesteps))

# Load your real-world image
real_image = Image.open("ddim_generated_image.png").convert("RGB").resize((256, 256))
real_image = torch.tensor(np.array(real_image)).float() / 255.0  # Normalize to [0, 1]
# real_image = (real_image * 2) - 1 # Normalize to [-1, 1] if required by the model
real_image = real_image.permute(2, 0, 1).unsqueeze(0).to(pipe.device)  # [B, C, H, W]


# Initialize latent state
x_t = real_image.clone()

# Reverse diffusion loop
count = 0
for t in reversed(timesteps):
    with torch.no_grad():
        # Predict the noise at this timestep
        noise_pred = pipe.unet(x_t, t)["sample"]

        # Calculate the noise to add
        alpha_t = alphas_cumprod[t] / 20
        noise = (x_t - (alpha_t ** 0.5) * noise_pred) / (1 - alpha_t) ** 0.5

        # Add the noise to go backwards
        x_t = x_t + noise  # Customize based on scheduler formula
        print(f"Inversion {t}/{len(timesteps)}, Shape: {x_t.shape}")

        # Save intermediate reverse step
        image = (x_t / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        generated_image = numpy_to_pil(image)[0]
        generated_image.save(f"step_{count:03d}.png")

        count += 1


# Latent representation after inversion
latent_noise = x_t
latent_state = latent_noise
print(f"Latent State Shape: {latent_state.shape}")

# Loop through timesteps in forward order
for t in pipe.scheduler.timesteps:
    with torch.no_grad():
        # Predict noise (model output)
        noise_pred = pipe.unet(latent_state, t)["sample"]

        # Use the scheduler to compute the next latent state
        output = pipe.scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latent_state,
            eta=0.0  # Typically set to 0.0 for deterministic results
        )

        # Update latent state
        latent_state = output.prev_sample

# Convert final latent state to an image
image = (latent_state / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()
generated_image = numpy_to_pil(image)[0]

# generated_image = latent_state.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
# generated_image = (generated_image * 255).astype(np.uint8)
# generated_image = Image.fromarray(generated_image)

# Save or show the generated image
generated_image.save("generated_from_latent.png")