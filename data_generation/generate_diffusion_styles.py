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
    num_inference_steps = 50
    n_samples = 1000
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

    print("Sampling latent codes in chunks...")

    # Load the model
    sd = load_model(
        model_id,
        device="cuda",
        h_space=h_space,
        num_inference_steps=num_inference_steps
    )

    # Sample and save chunks
    c = 0
    with tqdm.tqdm(total=n_samples, desc="Sampling latent codes") as pbar:
        for chunk_id in range(0, n_samples, chunk_size):
            chunk_file = os.path.join(chunk_folder, f"chunk_{chunk_id // chunk_size}.pkl")
            current_chunk = []
            for _ in range(min(chunk_size, n_samples - chunk_id)):
                q = sd.sample()
                current_chunk.append({
                    'hs': q.hs,
                    'x0': q.x0,
                    'w0': q.w0
                })
                pbar.update(1)  # Update the progress bar for each sample

                pil_image = tensor_to_pil(q.x0)[0]
                output_path = os.path.join(sample_images_folder, f"{c}.jpg")
                pil_image.save(output_path)
                c += 1

            # Save the current chunk
            with open(chunk_file, 'wb') as f:
                pickle.dump(current_chunk, f)

    # Consolidate all chunks into a single file
    print("Consolidating all chunks...")
    sampled_latent_codes = []
    for chunk_id in range(0, n_samples, chunk_size):
        chunk_file = os.path.join(chunk_folder, f"chunk_{chunk_id // chunk_size}.pkl")
        with open(chunk_file, 'rb') as f:
            current_chunk = pickle.load(f)
            sampled_latent_codes.extend(current_chunk)

    with open(style_file, 'wb') as f:
        pickle.dump(sampled_latent_codes, f)
    print(f"Consolidated file saved at {style_file}")

    # Remove chunk folder and its contents
    print("Cleaning up temporary chunk files...")
    shutil.rmtree(chunk_folder)
    print(f"Chunk folder '{chunk_folder}' removed.")

    # Example output for verification
    print(f"Number of sampled latent codes: {len(sampled_latent_codes)}")
    print(f"Sample structure: {sampled_latent_codes[0].keys() if sampled_latent_codes else 'No data'}")
