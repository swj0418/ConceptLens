from diffusion_utils.semanticdiffusion import load_model, Q
from diffusion_utils.direction_plotter import DirectionPlotter
from diffusion_utils.dutils import tensor_to_pil
import torch
import os
import pickle
from sklearn.cluster import AgglomerativeClustering
from PIL import Image


def process_images(folder_path, model_id, h_space, num_inference_steps, num_sampling_runs=64, walked_distance=10, inversion=False,
                   interfere_start=0, interfere_end=80):
    torch.manual_seed(42)

    style_file = f'styles/{model_id}-{num_inference_steps}_styles_500.pkl'  # Path to the style content file

    # Load the model
    sd = load_model(
        model_id,
        device="cuda",
        h_space=h_space,
        num_inference_steps=num_inference_steps
    )

    # Initialize DirectionPlotter
    dp = DirectionPlotter(sd)

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

    for i, item in enumerate(saved_latent_codes):
        print(i)

    codes = []
    for i, item in enumerate(saved_latent_codes):
        codes.append(item['hs'])
        # print(len(codes))
        saved_latent_codes[i] = 0
    codes = torch.stack(codes)

    # Cluster latent codes using Hierarchical Clustering
    shape = codes.shape  # [n_codes, latent_dim, timesteps, spatial, spatial] => [200, 896, 80, 8, 8]
    clustering = AgglomerativeClustering(n_clusters=32).fit(codes.flatten(start_dim=1).cpu().numpy())
    labels = clustering.labels_

    # Compute centroids for each cluster
    centroids = []
    for cluster_idx in range(32):
        cluster_points = codes[labels == cluster_idx]
        centroid = torch.mean(cluster_points, dim=0)
        centroids.append(centroid)
    centroids = torch.stack(centroids).to(codes.device)

    # Generate directions on-the-fly
    all_directions = []
    for run_idx in range(num_sampling_runs):
        print(f"Sampling run: {run_idx}")
        ws_a, ws_b = torch.randint(0, centroids.shape[0], (2,))
        while ws_a == ws_b:
            ws_b = torch.randint(0, centroids.shape[0], (1,))

        # Compute direction between centroid pairs
        direction = centroids[ws_a] - centroids[ws_b]
        direction = direction / torch.norm(direction, dim=-1, keepdim=True)
        direction = direction.reshape(shape[1:])
        all_directions.append(direction)

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
            direction[:interfere_start] = 0
            direction[interfere_end:] = 0
            q_edit = sd.apply_direction_raw(q.copy(), direction, scale=walked_distance)

            # Convert tensor to PIL image and save the edited image
            pil_image = tensor_to_pil(q_edit.x0)[0]
            output_path = os.path.join(walked_folder, f"{image_file.split('.')[0]}-{direction_idx}.jpg")
            pil_image.save(output_path)
            print(f"Processed and saved edited image: {output_path}")


if __name__ == '__main__':
    # input_folder = 'ldm_celeba256-vac-global-all'  # Replace with the path to your folder containing images
    #
    # process_images(input_folder, 'ldm', 'after', 80,
    #                walked_distance=25, num_sampling_runs=64, inversion=False)

    input_folder = 'ldm_celeba256-vac-global-middle_0'  # Replace with the path to your folder containing images

    process_images(input_folder, 'ldm', 'after', 80,
                   walked_distance=25, num_sampling_runs=64, inversion=False, interfere_start=20, interfere_end=35)

    input_folder = 'ldm_celeba256-vac-global-middle_1'  # Replace with the path to your folder containing images

    process_images(input_folder, 'ldm', 'after', 80,
                   walked_distance=25, num_sampling_runs=64, inversion=False, interfere_start=35, interfere_end=50)

    input_folder = 'ldm_celeba256-vac-global-late_0'  # Replace with the path to your folder containing images

    process_images(input_folder, 'ldm', 'after', 80,
                   walked_distance=25, num_sampling_runs=64, inversion=False, interfere_start=50, interfere_end=65)

    input_folder = 'ldm_celeba256-vac-global-late_1'  # Replace with the path to your folder containing images

    process_images(input_folder, 'ldm', 'after', 80,
                   walked_distance=25, num_sampling_runs=64, inversion=False, interfere_start=65, interfere_end=80)