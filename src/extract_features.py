import os
from src.data import dataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from data.dataset import SimpleISLESDataset

# Import your model
from models.networks.mmunetvae import MultiModalUNetVAE 



def extract_features(
    weights_path: str,
    data_dir: str,
    output_csv: str,
    batch_size: int = 1
):
    print("Loading MM-UNet-VAE model and weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the foundation model architecture
    model = MultiModalUNetVAE(
        input_channels=2,
        output_channels=1,
        starting_filters=32,
    )
    
    # Load the pre-trained weights
    state_dict = torch.load(weights_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    clean_state_dict = {k.replace('model.', '').replace('network.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Setting up custom dataset from {data_dir}...")
    # USE THE NEW CUSTOM DATASET HERE
    dataset = SimpleISLESDataset(data_dir) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Global Average Pooling to flatten 3D maps to 1D arrays
    pooler = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    extracted_features = []
    subject_ids = []

    print(f"Extracting 768-dimensional latent representations for {len(dataset)} subjects...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)  # [1, 2, X, Y, Z]
            sub_ids = batch['subject_id']
            
            z_s_all = []
            z_m_all = []

            # Process Modality 0 (ADC) then Modality 1 (DWI) individually
            for ix in range(model.num_modalities):
                # Slicing with [ix] preserves shape as [1, 1, X, Y, Z]
                x_mod = images[:, [ix]] 
                
                # 1. Pass single modality through the 1-channel encoder
                skips = model.encoder(x_mod)
                
                # 2. Replicate the foundation model's multiscale feature pooling
                bottleneck_shape = skips[-1].shape[2:]
                multiscale_feats = [F.adaptive_avg_pool3d(f, output_size=bottleneck_shape) for f in skips]
                pooled_bottleneck = torch.cat(multiscale_feats, dim=1)
                
                # 3. Project into the VAE latent spaces
                mu_s = model.conv_mu_shared(pooled_bottleneck)
                mu_m = model.conv_mu_modality(pooled_bottleneck)
                
                z_s_all.append(mu_s)
                z_m_all.append(mu_m)

            # 4. Replicate paper's fusion: Average the shared Subject anatomy
            z_s_fused = torch.stack(z_s_all).mean(dim=0)
            
            # 5. Assemble the ultimate patient vector: [Anatomy, ADC_physics, DWI_physics]
            # (This is the exact tensor the authors feed into their classification heads!)
            full_patient_tensor = torch.cat([z_s_fused, z_m_all[0], z_m_all[1]], dim=1)

            # 6. Global Average Pool the remaining 3D spatial grid down to a flat 1D vector
            patient_vector = pooler(full_patient_tensor).squeeze(-1).squeeze(-1).squeeze(-1)
            
            extracted_features.append(patient_vector.cpu().numpy())
            subject_ids.extend(sub_ids)

    # Format into a Pandas DataFrame and save
    print("Formatting CSV...")
    feature_matrix = np.vstack(extracted_features)
    num_features = feature_matrix.shape[1]
    col_names = [f"feat_{i}" for i in range(num_features)]
    
    df = pd.DataFrame(feature_matrix, columns=col_names)
    df.insert(0, "Subject_ID", subject_ids)
    
    df.to_csv(output_csv, index=False)
    print(f"Success! Features for {len(subject_ids)} subjects saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to your .ckpt or .pth file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed ISLES24 data")
    parser.add_argument("--out", type=str, default="isles24_features.csv", help="Output CSV filename")
    args = parser.parse_args()
    
    extract_features(args.weights, args.data_dir, args.out)