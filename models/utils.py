# -*- coding: UTF-8 -*-
# Local modules
import os
from typing import Dict, Union
# 3rd party modules
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.tensorboard import SummaryWriter
# Self-written modules
from models.dataset import TimeGANDataset
import warnings
from sklearn.covariance import ledoit_wolf

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.timegan import TimeGAN
#------------------------------------------------------------------------------------------------
def embedding_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    emb_epochs: int, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the embedding and recovery functions
    """  
    logger = trange(emb_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:   
        for X_mb, T_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")
            loss = np.sqrt(E_loss_T0.item())

            # Backward Pass
            E_loss0.backward()

            # Update model parameters
            e_opt.step()
            r_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Embedding/Loss:", 
                loss, 
                epoch
            )
            writer.flush()

def supervisor_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    sup_epochs: int, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the supervisor function
    """
    logger = trange(sup_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            S_loss = model(X=X_mb, T=T_mb, Z=None, obj="supervisor")

            # Backward Pass
            S_loss.backward()
            loss = np.sqrt(S_loss.item())

            # Update model parameters
            s_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Supervisor/Loss:", 
                loss, 
                epoch
            )
            writer.flush()

def joint_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    d_opt: torch.optim.Optimizer, 
    sup_epochs: int, 
    batch_size: int,
    max_seq_len: int,
    Z_dim: int,
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None, 
) -> None:
    """The training loop for training the model altogether
    """
    logger = trange(
        sup_epochs, 
        desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
    )
    
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            ## Generator Training
            for _ in range(2):
                # Random Generator
                Z_mb = torch.rand((batch_size, max_seq_len, Z_dim))

                # Forward Pass (Generator)
                model.zero_grad()
                G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                # Update model parameters
                g_opt.step()
                s_opt.step()

                # Forward Pass (Embedding)
                model.zero_grad()
                E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())
                
                # Update model parameters
                e_opt.step()
                r_opt.step()

            # Random Generator
            Z_mb = torch.rand((batch_size, max_seq_len, Z_dim))

            ## Discriminator Training
            model.zero_grad()
            # Forward Pass
            D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")

            # Check Discriminator loss
            # if D_loss > dis_thresh:
            # Backward Pass
            D_loss.backward()

            # Update model parameters
            d_opt.step()
            D_loss = D_loss.item()

        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
        )
        if writer:
            writer.add_scalar(
                'Joint/Embedding_Loss:', 
                E_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Generator_Loss:', 
                G_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Discriminator_Loss:', 
                D_loss, 
                epoch
            )
            writer.flush()

def timegan_trainer(model, dataset, p, max_seq_len, sup_epochs, emb_epochs, batch_size, device, learning_rate):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False    
    )

    model.to(device)

    # Initialize Optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=learning_rate)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=learning_rate)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=learning_rate)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"tensorboard"))

    print("\nStart Embedding Network Training")
    embedding_trainer(
        model=model, 
        dataloader=dataloader, 
        e_opt=e_opt, 
        r_opt=r_opt,
        emb_epochs=emb_epochs,  
        writer=writer
    )

    print("\nStart Training with Supervised Loss Only")
    supervisor_trainer(
        model=model,
        dataloader=dataloader,
        s_opt=s_opt,
        g_opt=g_opt,
        sup_epochs=sup_epochs,
        writer=writer
    )

    print("\nStart Joint Training")
    joint_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        sup_epochs=sup_epochs, 
        batch_size=batch_size,
        max_seq_len = max_seq_len,
        Z_dim=p,
        writer=writer,
    )


def timegan_generator(model, device, T, max_seq_len, Z_dim):
    """
    The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    
    
    # Initialize model to evaluation mode and run without gradients
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((len(T), max_seq_len, Z_dim))
        
        generated_data = model(X=None, T=T, Z=Z, obj="inference")
    
    return generated_data.numpy()

def create_rolling_windows(data, window_length, overlap):
    if window_length > data.shape[0]:
        warnings.warn(f'Caution! the window length  are more than the input timesteps. Adjusting accordingly!', UserWarning)
        window_length = data.shape[0]
        if overlap > window_length:
            warnings.warn(f'Caution! overlap was more than input timesteps. Adjusting accordingly!', UserWarning)
            overlap = 1

    step_size = window_length - overlap
    n_windows = (data.shape[0] - window_length) // step_size + 1

    windows = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_windows, window_length, data.shape[1]),
        strides=(step_size * data.strides[0], data.strides[0], data.strides[1])
    )
    return windows

def reconstruct_timeseries(windows, window_length, overlap):
    step_size = window_length - overlap
    n_windows = windows.shape[0]
    n_features = windows.shape[2]

    # Calculate the total length of the reconstructed time series
    reconstructed_length = (n_windows - 1) * step_size + window_length

    # Initialize an array to store the reconstructed time series and an array for the count of overlaps
    reconstructed_data = np.zeros((reconstructed_length, n_features))
    overlap_count = np.zeros((reconstructed_length, n_features))

    # Add windows to the reconstructed data, while counting the overlaps
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        reconstructed_data[start_idx:end_idx, :] += windows[i, :, :]
        overlap_count[start_idx:end_idx, :] += 1

    # Avoid division by zero by ensuring no element in overlap_count is zero
    overlap_count[overlap_count == 0] = 1

    # Calculate the mean of overlapping elements
    reconstructed_data /= overlap_count

    return reconstructed_data


def is_size_fixed(data, window_length, p):
    x1 = data.shape[0]
    x2 = data.shape[1]
    if x1 == window_length and x2 == p:
        return True
    else:
        return False

def pad_arrays(arrays, max_seq_len, padding_value):
    
    # Create a list to store the padded arrays
    padded_arrays = []
    
    # Pad each array as needed
    for array in arrays:
        # Calculate the number of rows to pad
        pad_height = max_seq_len - array.shape[0]
        
        # Perform the padding
        if pad_height > 0:
            padded_array = np.pad(array, ((0, pad_height), (0, 0)), mode='constant', constant_values=padding_value)
        else:
            padded_array = array
        
        # Add the padded array to the list
        padded_arrays.append(padded_array)
    
    return padded_arrays

def reimannian_mapping(data_list):
    '''
    data_list: list of timeseries datapoints
    '''
    # this function only works when given a list of timeseries and return the corresponding mapped covariance matrix

    Covs = []
    for i in range(len(data_list)):
        baseline = data_list[i]
        Cov = ledoit_wolf(baseline)[0]
        Covs.append(Cov)

    #Initialize the Matrix Mean:
    Covs = np.array(Covs)
    geomean = np.mean(Covs,axis = 0)
    #Initialize the gradient descent step size and loss:
    step = 1
    norm_old = np.inf

    #Set tolerance:
    tol = 1e-8
    norms = []

    for n in range(100):

        #Compute the gradient
        geo_eval,geo_evec = np.linalg.eigh(geomean)
        geomean_inv_sqrt = mat_op(np.sqrt,1. / geo_eval,geo_evec)

        #Project matrices to tangent space and compute mean and norm:
        mats= [geomean_inv_sqrt.dot(cov).dot(geomean_inv_sqrt) for cov in Covs]
        log_mats = [logarithm(mat) for mat in mats]
        meanlog = np.mean(log_mats,axis = 0)
        norm = np.linalg.norm(meanlog)

        #Take step along identified geodesic to minimize loss:
        geomean_sqrt = sqrroot(geomean)
        geomean = geomean_sqrt.dot(expstep(meanlog,step)).dot(geomean_sqrt)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm

        elif norm > norm_old:
            step = step / 2.
            norm = norm_old

        if tol is not None and norm / geomean.size < tol:
            break
            
        norms.append(norm)

    geo_eval,geo_evec = np.linalg.eigh(geomean)

    geomean_inv_sqrt = mat_op(np.sqrt,1. / geo_eval,geo_evec)

    T_covs = [T_Project(geomean_inv_sqrt,cov) for cov in Covs]

    print(f'{len(T_covs)} covariances are mapped')

    return T_covs

def discrimination_score(X, X_gen, T, method):

    if method == 'svm':
        # data preprocessing
        X = unpad_arrays(X, T)
        X_gen = unpad_arrays(X_gen, T)
        X_total = X + X_gen
        mapped_covs = reimannian_mapping(X_total)
        flat_covs = [cov.flatten() for cov in mapped_covs]

        # these mapped covariances are gonna be used for the discriminative analysis
        y = []
        for i in range(len(X)):
            y.append(1)
        for i in range(len(X_gen)):
            y.append(0)

        X_train, X_test, y_train, y_test = train_test_split(flat_covs,y, test_size = 0.2)
        eval_model = SVC(kernel='rbf')
        eval_model.fit(X_train, y_train)
        y_hat = eval_model.predict(X_test)
        dis_score = accuracy_score(y_true=y_test, y_pred = y_hat)
        
        return dis_score

    elif method == 'lstm':
        return None

def logarithm(cov):
    d, V = np.linalg.eigh(cov)
    D = np.diag(np.log(d))
    logcov = np.dot(np.dot(V, D), V.T)
    return logcov

def sqrroot(cov):
    d, V = np.linalg.eigh(cov)
    D = np.diag(np.sqrt(d))
    sqrroot = np.dot(np.dot(V, D), V.T)
    return sqrroot

def expstep(cov,step):
    d, V = np.linalg.eigh(cov)
    D = np.diag(np.exp(d*step))
    expstep = np.dot(np.dot(V, D), V.T)
    return expstep

def mat_op(operation,d,V):
    return np.dot(V*operation(d),V.T)

def T_Project(geomean_inv_sqrt,cov):
        newmat = geomean_inv_sqrt.dot(cov).dot(geomean_inv_sqrt)
        T_cov = logarithm(newmat)
        return T_cov

def TimeGAN_trainer(device, dataset, p, max_seq_len, project_filepath):

    model = TimeGAN(device=device, feature_dim=p, Z_dim=p, hidden_dim=512, max_seq_len=max_seq_len, batch_size=32, num_layers=3, padding_value=100)
    timegan_trainer(model=model, dataset=dataset, batch_size=32, device=device, learning_rate=0.0002, p=p, max_seq_len=max_seq_len, sup_epochs=100, emb_epochs=100)
    return model