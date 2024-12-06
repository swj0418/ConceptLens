from diffusion_utils.semanticdiffusion import load_model, Q
import tqdm
import torch
import os
import pickle
import shutil
from diffusion_utils.dutils import tensor_to_pil


if __name__ == '__main__':
    model_id = 'ldm'
    h_space = 'after'
    num_inference_steps = 80
    n_samples = 500
    chunk_size = 50  # Number of samples per chunk

    output_root = 'styles'
    os.makedirs(output_root, exist_ok=True)
    sample_images_folder = os.path.join(output_root, 'diffusion_sample_images')

    # Create a subfolder for chunk files
    chunk_folder = os.path.join(output_root, f"{model_id}_chunks")
    os.makedirs(chunk_folder, exist_ok=True)
    os.makedirs(sample_images_folder, exist_ok=True)

    # Final consolidated file
    style_file = os.path.join(output_root, f"{model_id}-{num_inference_steps}_styles_{n_samples}.pkl")

    # Consolidate all chunks into a single file
    print("Consolidating all chunks...")
    sampled_latent_codes = []
    for chunk_id in range(0, n_samples, chunk_size):
        print(chunk_id // chunk_size)
        chunk_file = os.path.join(chunk_folder, f"chunk_{chunk_id // chunk_size}.pkl")
        with open(chunk_file, 'rb') as f:
            current_chunk = pickle.load(f)
            newcc = []
            for item in current_chunk:
                tmp = {}
                for k, v in item.items():
                    tmp[k] = v.detach().cpu()
                newcc.append(tmp)
            sampled_latent_codes.extend(newcc)

        if chunk_id // chunk_size == 4:
            break

    with open(style_file, 'wb') as f:
        pickle.dump(sampled_latent_codes, f)
    print(f"Consolidated file saved at {style_file}")

    # Example output for verification
    print(f"Number of sampled latent codes: {len(sampled_latent_codes)}")
    print(f"Sample structure: {sampled_latent_codes[0].keys() if sampled_latent_codes else 'No data'}")
