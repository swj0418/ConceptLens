from diffusion_utils.semanticdiffusion import load_model, Q
from diffusion_utils.direction_plotter import DirectionPlotter
from diffusion_utils.dutils import tensor_to_pil
from diffusion_utils.pca import PCAMethod
import torch
import os
import pickle
from sklearn.cluster import AgglomerativeClustering
from PIL import Image


def process_images(folder_path, model_id, h_space, num_inference_steps, num_sampling_runs=64, walked_distance=10, inversion=False):
    torch.manual_seed(42)

    style_file = f'styles/{model_id}-{h_space}-{num_inference_steps}_styles_200.pkl'  # Path to the style content file

    # Load the model
    sd = load_model(
        model_id,
        device="cuda",
        h_space=h_space,
        num_inference_steps=num_inference_steps
    )

    # Collect all images from the specified folder
    image_files = [f for f in os.listdir(os.path.join(folder_path, 'codes')) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort based on image numbering

    # Create output directories
    walked_folder = os.path.join(folder_path, "walked")
    inverted_folder = os.path.join(folder_path, "inverted")
    os.makedirs(walked_folder, exist_ok=True)
    os.makedirs(inverted_folder, exist_ok=True)

    # Load latent codes ("hs") from the style file
    with open(style_file, "rb") as f:
        saved_latent_codes = pickle.load(f)
    codes = []
    for item in saved_latent_codes:
        codes.append(item['hs'])
    codes = torch.stack(codes)

    # Generate directions on-the-fly
    pca = PCAMethod(sd)
    dp = DirectionPlotter(sd)
    all_directions = []
    for run_idx in range(num_sampling_runs):
        hhs = codes[run_idx*25:(run_idx+1)*25]
        PCs, ss, Uts = pca.get_PCs_indv(hhs, num_svectors=10)
        for i in range(8):
            n = dp.get_direction(Uts, ss, svec_idx=i)
            all_directions.append(n.delta_hs)

    latent_codes = []
    # Collect latent codes from all images
    if os.path.exists(os.path.join(folder_path, 'codes.pt')):
        latent_codes = torch.load(os.path.join(folder_path, 'codes.pt'))
    else:
        for i, image_file in enumerate(image_files):
            print(f"Encoding original codes: {i}")
            image_path = os.path.join(folder_path, 'codes', image_file)
            x0 = sd.img_path_to_tensor(image_path)

            # Create a Q object and encode the image
            q = Q(x0=x0, etas=0)
            q = sd.encode(q)
            latent_codes.append(q)

        torch.save(latent_codes, os.path.join(folder_path, 'codes.pt'))

    # Save the reconstructed images
    if inversion:
        for i, q in enumerate(latent_codes):
            print(f"Inverting original codes: {i}")
            q = sd.decode(q)
            pil_image = tensor_to_pil(q.x0)[0]
            output_path = os.path.join(inverted_folder, f"{i}.jpg")
            pil_image.save(output_path)

    # Apply edits using the computed directions
    for idx, (image_file, q) in enumerate(zip(image_files, latent_codes)):
        for direction_idx, direction in enumerate(all_directions): # [:30] looked good
            # direction[80:90] *= 3
            # direction[60:] = 0
            q_edit = sd.apply_direction_raw(q.copy(), direction, scale=walked_distance)

            # Convert tensor to PIL image and save the edited image
            pil_image = tensor_to_pil(q_edit.x0)[0]
            output_path = os.path.join(walked_folder, f"{image_file.split('.')[0]}-{direction_idx}.jpg")
            pil_image.save(output_path)
            print(f"Processed and saved edited image: {output_path}")


if __name__ == '__main__':
    input_folder = 'ldm_celeba256-vac-global-all'  # Replace with the path to your folder containing images

    process_images(input_folder, 'ldm', 'after', 80,
                   walked_distance=10, num_sampling_runs=8, inversion=False)
