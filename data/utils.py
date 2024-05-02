import numpy as np
import pandas as pd 
import pickle 
import os 
import torch
from torch.utils.data import TensorDataset
#------------------------------------------------------------------------------------------------------------
def normalizer(datalist):
    # Assuming your_data is a NumPy array of shape (n_timesteps, n_datastreams)
    means = datalist.mean(axis=0)
    stds = datalist.std(axis=0)

    normalized_data = (datalist - means) / stds

    return normalized_data

def normalize_data(data_to_normalize, reference_data):
    means = np.mean(reference_data, axis=0)
    stds = np.std(reference_data, axis=0)
    # Check for any zero standard deviations and raise an error
    if np.any(stds == 0):
        raise ValueError("Zero standard deviation encountered, which can lead to division by zero errors.")
        
    normalized_data = (data_to_normalize - means) / stds
    return normalized_data

def dataloader(project_filepath, normalize=True):
    '''
    Loads in-control and out-of-control data
    project_filepath: main folder path that containts all files of the project
    datatype: a string that shows which type of data is going to be loaded, 'ic' for in-control 'oc' for out-of-control

    The ic data will be normalized after loading. This is not the case for OC, OC data is later normalized in training of SPC engine.
    '''

    data_filepath = f'{project_filepath}IC and OC data/'
    
    if normalize:
        with open(f'{data_filepath}ic_list.pkl', 'rb') as file:
            ic_list = pickle.load(file)

        with open(f'{data_filepath}oc_list.pkl', 'rb') as file:
            oc_list = pickle.load(file)

        p = ic_list[0].shape[1]

        print(f'{p} number of datastreams are monitored!')

        # Normalize OC data based on IC data
        oc_list_normalized = []
        data_ic = ic_list[0]
        for i, oc_data in enumerate(oc_list):   
            for j in range(p):
                data_j = data_ic[:, j]
                std_j = data_j.std()
                mean_j = data_j.mean()
                oc_data[:, j] = (oc_data[:, j] - mean_j) / std_j
            oc_list_normalized.append(oc_data)

        
        # If you need to normalize IC data as well
        ic_list_normalized = [normalize_data(ic, ic) for ic in ic_list]

        return ic_list_normalized, oc_list_normalized
    elif normalize == False:
        print('code not completed')

def train_test_split(datalist, labels, datatype, test_size, num_sim, n_faults):
    
    L = len(datalist)
    len_test = int(test_size * num_sim)
    len_train = int(num_sim - len_test)
    train_list = []
    test_list = []
    y_train = []
    y_test = []
    
    if datatype == 'oc':
        for fault in range(n_faults):

            oc_data = datalist[(fault)*num_sim : (fault + 1)*num_sim]
            print(f'len fault {fault+1} dataset = {len(oc_data)}')
            train_list += oc_data[:len_train]
            test_list += oc_data[len_train:]
            labels_f = labels[(fault)*num_sim : (fault + 1)*num_sim]
            y_train += labels_f[:len_train]
            y_test += labels_f[len_train:]
    
        return train_list, test_list, y_train, y_test

    elif datatype == 'ic':
        train_list += datalist[:len_train]
        test_list += datalist[len_train:]
    
        return train_list, test_list

def save_file(file,file_name,file_path, file_format):
    '''
    Saves variables as files based on the file format

    fileformat: eligible file formats are:
        - 'npy' for numpy arrays
        - 'pkl; for pickle (useful for lists)
        - 'csv' for saving pandas dataframes as csv files

    '''
    if not os.path.exists(file_path):
            os.makedirs(file_path)

    if file_format == 'pkl':
        
        with open(f'{file_path}{file_name}.pkl', 'wb') as fileee:
            pickle.dump(file, fileee)

def truncate(fault_number, oc_data, after_RL_window, project_filepath):
    '''
    Truncating the timeseries based on the obtained RLs in Phase II
    - fault_number: fault's label
    - oc_data: list of simulation timeseries for fault_number
    - after_RL_window: additional number of timesteps to consider after RL (to capture more information)
    '''
    load_path = f'{project_filepath}/Phase II analysis/Multivariate/Run_Lengths'
    truncated_oc_data = []
    # Loading fault's RL file list

    with open(f'{load_path}/RL_f{fault_number}.pkl', 'rb') as file:
        RL_f = pickle.load(file)
        
    for i, timeseries in enumerate(oc_data):
        RL_i = RL_f[i]
        truncation_time_length = RL_i + after_RL_window
        truncated_timeseries = timeseries[:truncation_time_length,:]
        truncated_oc_data.append(truncated_timeseries)

    return truncated_oc_data, after_RL_window

def segment_time_series(data, window_length):
    segments = []
    start = 0
    
    # Iterate through data in steps of window_length
    while start < len(data):
        # Check if the remaining data is less than the window length
        if start + window_length >= len(data):
            # Take all remaining data
            segments.append(data[start:])
            break
        else:
            # Take a slice of data of window length
            segments.append(data[start:start + window_length])
        start += window_length
    
    return segments

def reconstruct_time_series(segments):
    reconstructed_data = []
    for segment in segments:
        reconstructed_data.extend(segment)
    return reconstructed_data

def unpad_arrays(padded_arrays, original_row_counts):
    unpadded_arrays = []
    
    # Iterate over each padded array and its corresponding original row count
    for padded_array, original_rows in zip(padded_arrays, original_row_counts):
        # Slice the array to its original number of rows
        unpadded_array = padded_array[:original_rows, :]
        unpadded_arrays.append(unpadded_array)
        
    return unpadded_arrays

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data

def dataset_maker(data_name, seq_len):
    
    # Loading data 

    data = real_data_loading(data_name, seq_len)
    

    X = list()
    T = list()

    for i, sim_data in enumerate(data):
            X.append(sim_data)
            T.append(sim_data.shape[0])
    # X is time series dataset as a PyTorch Tensor of shape [num_samples, seq_length, features]
    # T is a list of timesteps for each sample

    X = np.array(X)
    X = torch.tensor(X, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.int)

    # Check the shape of the tensor 
    print(f'Processing training data with shape {X.shape}')
    
    # Making a TorchDataset with X and T (models work with both)
    dataset = TensorDataset(X,T) 

    return X, T, dataset