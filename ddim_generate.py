from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("merkol/ddim-ema-ffhq-256")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
