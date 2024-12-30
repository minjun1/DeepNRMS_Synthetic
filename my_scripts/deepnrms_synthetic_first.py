"""
deepnrms_synthetic_first.py

Core functionality for a synthetic example of time-lapse seismic monitoring for CCS:
1) Load pre-/post-injection seismic data.
2) Create time-lapse differences and normalize them.
3) Extract 2D patches.
4) Load a pre-trained CNN/encoder (autoencoder).
5) Fine-tune or optimize the encoder with a distance-based objective (Deep SVDD).
6) Generate anomaly scores on post-injection data.

Author: [Your Name]
Date: [Date]
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim


# Example: This might be your CNN model definition file
#import cae_ccus      # If you had a CAE architecture
import cae_128x128_up as cae  # Another CAE architecture you used
import deepnrms    # If you have a separate deep SVDD class

def load_pre_post_data(pre_path, post_path):
    """
    Load pre-injection and post-injection .npy files.
    Returns: images_pre, images_post
    """
    images_pre = np.load(pre_path)  # shape: (N_pre+1, nx, nz) for example
    images_post = np.load(post_path)
    return images_pre, images_post

def create_time_lapse_diff(images_pre, images_post):
    """
    Create difference images:
    - For each time index i in 'pre', diff_pre[i] = pre[i+1] - pre[0]
    - For each time index i in 'post', diff_post[i] = post[i] - pre[0]

    Parameters
    ----------
    images_pre : np.ndarray (shape [N_pre, nx, nz])
    images_post : np.ndarray (shape [N_post, nx, nz])

    Returns
    -------
    image_diff_pre : np.ndarray
    image_diff_post : np.ndarray
    """
    n_pre = images_pre.shape[0]
    n_post = images_post.shape[0]

    # For example, first index is baseline
    image_diff_pre = np.zeros((n_pre - 1, images_pre.shape[1], images_pre.shape[2]))
    image_diff_post = np.zeros((n_post, images_post.shape[1], images_post.shape[2]))

    for i in range(n_pre - 1):
        image_diff_pre[i] = images_pre[i + 1] - images_pre[0]

    for i in range(n_post):
        image_diff_post[i] = images_post[i] - images_pre[0]

    return image_diff_pre, image_diff_post

def clip_and_normalize(image_diff, depth_windowing=450):
    """
    Clip at 99th percentile and normalize by max absolute value, only 
    in the bottom 'depth_windowing' portion along z-axis.
    Returns a new array shape [n_slices, nx, depth_windowing].
    """
    n_slices, nx, nz = image_diff.shape
    data_norm = np.zeros((n_slices, nx, depth_windowing))
    for i in range(n_slices):
        slice_i = image_diff[i, :, -depth_windowing:]  # focus on deeper zone
        clip_val = np.percentile(slice_i, 99.0)
        slice_i = np.clip(slice_i, -clip_val, clip_val)
        maxabs = np.max(np.abs(slice_i))
        if maxabs > 0:
            slice_i = slice_i / maxabs
        data_norm[i] = slice_i
    return data_norm

def extract_patches_2d(data_2d, patch_size=128, slide_x=10, slide_z=10):
    """
    Extract overlapping 2D patches from a single 2D array data_2d (nx, nz).
    Returns a list/array of patches each (patch_size, patch_size).
    """
    nx, nz = data_2d.shape
    patches = []
    for ix in range(0, nx - patch_size, slide_x):
        for iz in range(0, nz - patch_size, slide_z):
            patch = data_2d[ix:ix+patch_size, iz:iz+patch_size]
            patches.append(patch)
    return np.array(patches)

def build_dataset_from_slices(data_norm, patch_size=128, slide_x=10, slide_z=10):
    """
    Combine patches from multiple slices into one dataset.
    data_norm : shape [n_slices, nx, depth_windowing]
    Returns: data array shape (N_total, 1, patch_size, patch_size)
    """
    all_patches = []
    n_slices = data_norm.shape[0]
    for i in range(n_slices):
        patches_i = extract_patches_2d(data_norm[i], patch_size, slide_x, slide_z)
        all_patches.append(patches_i)
    data_combined = np.concatenate(all_patches, axis=0)
    # Reshape to (N, 1, patch_size, patch_size) for PyTorch
    data_combined = np.reshape(data_combined, (data_combined.shape[0], 1, patch_size, patch_size))
    return data_combined


def train_autoencoder(
    data_train,
    cae_model_class,  # We'll assume you pass in your CAE class
    num_epochs=50,
    batch_size=2048,
    model_save_path='model_ae_best.pth',
    device='cuda'
):
    """
    Train a CNN Autoencoder on the data_train patches.

    data_train : np.ndarray, shape (N, 1, patch_size, patch_size)
    cae_model_class : class or callable
        The CAE class (e.g., cae.CAE) to instantiate the model.
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for DataLoader
    model_save_path : str
        Path where the best model weights will be saved
    device : str, 'cuda' or 'cpu'
        The device to train on

    Returns
    -------
    model : torch.nn.Module
        The trained model with best weights loaded.
    """

    # Initialize CAE model
    model = cae_model_class()  # e.g., cae.CAE()
    model = model.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Create DataLoader
    train_dataset = torch.FloatTensor(data_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dataset_size = len(train_dataset)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(inputs, outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Compute average loss for this epoch
        epoch_loss = running_loss / dataset_size
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.6f}")

        # Check if this is the best (lowest) loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)
            print(f"  (New best model - saved to {model_save_path})")

    # Load the best weights into the model
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best loss: {best_loss:.6f}")

    return model

def load_pretrained_encoder(model_path):
    """
    Load a pre-trained CAE model, return the encoder part.
    """
    model = cae.CAE()
    model.load_state_dict(torch.load(model_path))
    encoder = model.encoder
    # In some setups, you might do: encoder = copy.deepcopy(model.encoder)
    return encoder

def deep_svdd_loss(embeddings, center, eps=1e-4):
    """
    Compute the DeepNRMS loss:
    mean( sum( (z_i - c)**2 ) ), where z_i is the embedding.
    'center' is a vector with same dimension as embeddings.
    """
    dist = (embeddings - center) ** 2
    dist_sum = torch.sum(dist, dim=1)  # sum over embedding dim
    loss_val = torch.mean(dist_sum)
    return loss_val

def compute_center(encoder, data_tensor, device='cuda', eps=0.1):
    """
    Pass data through encoder, compute mean embedding => center c.
    Then apply the typical rule to avoid zero dimension:
    if abs(c_i) < eps, c_i = +/- eps
    """
    encoder.eval()
    embeddings_list = []
    with torch.no_grad():
        for x in data_tensor:
            x = x.unsqueeze(0).to(device)  # shape (1, 1, H, W)
            z = encoder(x)
            embeddings_list.append(z.cpu())
    embeddings = torch.cat(embeddings_list, dim=0)  # shape (N, z_dim)
    c = torch.mean(embeddings, dim=0)
    # clamp small absolute values to +/- eps
    c[(torch.abs(c) < eps) & (c < 0)] = -eps
    c[(torch.abs(c) < eps) & (c > 0)] =  eps
    return c

import copy
import torch

def optimize_encoder(
    encoder,
    data_tensor,
    center,
    device='cuda',
    lr=1e-4,
    weight_decay=5e-7,
    num_epochs=30,
    best_model_path=None
):
    """
    Fine-tune the encoder using a distance-based objective (Deep SVDD style):
      mean( (z - center)^2 ), for z = encoder(x).

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder model to be optimized.
    data_tensor : torch.Tensor
        A 4D tensor of shape (N, 1, H, W) containing training data patches.
    center : torch.Tensor
        The center vector c (must be same device as the model).
    device : str
        'cuda' or 'cpu'.
    lr : float
        Learning rate for Adam.
    weight_decay : float
        L2 regularization factor.
    num_epochs : int
        Number of epochs for fine-tuning.
    best_model_path : str or None
        If provided, we will save the best model weights to this file path.

    Returns
    -------
    encoder : torch.nn.Module
        The optimized encoder with the best weights loaded.
    """

    # Ensure encoder and center are on same device
    encoder = encoder.to(device)
    center = center.to(device)

    # Create a DataLoader
    train_loader = torch.utils.data.DataLoader(
        data_tensor, 
        batch_size=4096, 
        shuffle=True
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(
        encoder.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(encoder.state_dict())

    for epoch in range(num_epochs):
        encoder.train()
        running_loss = 0.0

        for x in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            z = encoder(x)
            loss = torch.mean(torch.sum((z - center) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.6f}")

        # Check for best (lowest) loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(encoder.state_dict())
            if best_model_path is not None:
                torch.save(best_model_wts, best_model_path)
                print(f"  (Best model so far - saved to {best_model_path})")

    # Load the best weights
    encoder.load_state_dict(best_model_wts)
    print(f"Finished training. Best loss: {best_loss:.6f}")
    return encoder


def assemble_patches_score(scores, patch_size=128, slide_x=10, slide_z=10, 
                           nx=1000, nz=450):
    """
    Reconstruct a 'heatmap' from the list of patch scores.
    'scores' is a list with length = number_of_patches
    'heatmap' accumulates the sum, while 'div' accumulates how many times a pixel is visited.
    Return the average heatmap = heatmap / div.
    """
    heatmap = np.zeros((nx, nz))
    div = np.zeros((nx, nz))
    count_idx = 0
    for ix in range(0, nx - patch_size, slide_x):
        for iz in range(0, nz - patch_size, slide_z):
            heatmap[ix:ix+patch_size, iz:iz+patch_size] += scores[count_idx]
            div[ix:ix+patch_size, iz:iz+patch_size] += 1
            count_idx += 1
    div[div == 0] = 1
    return heatmap / div

def load_optimized_encoder(encoder_class, checkpoint_path, device='cuda'):
    """
    Create a new encoder model from 'encoder_class', load the best weights from
    'checkpoint_path', move to 'device', and set eval mode.

    Parameters
    ----------
    encoder_class : class
        The class or callable that creates an encoder (e.g., cae.CAE).
    checkpoint_path : str
        Path to the .pth file containing the model.state_dict().
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    encoder : torch.nn.Module
        The newly created encoder with loaded weights, on 'device', in eval mode.
    """

    # 1. Instantiate a new model object
    encoder = encoder_class()
    # 2. Load state dict from checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    # 3. Move the model to the specified device
    encoder = encoder.to(device)
    # 4. Set evaluation mode
    encoder.eval()

    return encoder

def compute_anomaly_score(encoder, center, data_2d, patch_size=128, slide_x=10, slide_z=10, device='cuda'):
    """
    For a single 2D array (nx, nz), extract patches, pass through encoder,
    compute distance^2 to 'center' for each patch => average as the patch score.
    Return a 2D heatmap of anomaly scores.
    """
    patches = extract_patches_2d(data_2d, patch_size=patch_size, slide_x=slide_x, slide_z=slide_z)
    if patches.shape[0] == 0:
        return np.zeros_like(data_2d)

    # shape (N_patches, 1, patch_size, patch_size)
    patches = np.reshape(patches, (patches.shape[0], 1, patch_size, patch_size))
    data_tensor = torch.FloatTensor(patches).to(device)

    # Ensure center is on the same device as the encoder/data
    center = center.to(device)

    encoder.eval()
    scores = []
    with torch.no_grad():
        for x in data_tensor:
            # shape: (1, 1, patch_size, patch_size) => encoder input
            x = x.unsqueeze(0)  
            z = encoder(x)      # shape: (1, z_dim)
            # Now both z and center are on the same device
            dist_sq = torch.sum((z - center) ** 2, dim=1)
            scores.append(dist_sq.item())

    # Reassemble patches back into a map
    heatmap = assemble_patches_score(
        scores, 
        patch_size=patch_size, 
        slide_x=slide_x, 
        slide_z=slide_z,
        nx=data_2d.shape[0], 
        nz=data_2d.shape[1]
    )
    return heatmap
