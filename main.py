import torch
import os
import numpy as np
from models.utils import timegan_generator
from data.utils import unpad_arrays, save_file, segment_time_series, dataset_maker
from models.utils import TimeGAN_trainer

if __name__ == "__main__":

    project_filepath = os.path.abspath(".")

    if not os.path.exists(project_filepath):
        raise ValueError(f"Code directory not found at {project_filepath}.")
    
    intended_faults = [i for i in range(1,21)]
    seed = 42
    if torch.cuda.is_available():
        print("Using CUDA\n")
        device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        device = torch.device("cpu")
    seq_len = 24

    X, T, dataset = dataset_maker(data_name='stock', seq_len=seq_len)
    p = X.shape[2]
    # # TODO: Save the original data, modify the code first
    save_file(file=X.numpy(), file_format='pkl', file_name='original', file_path=f'{project_filepath}/data/')
    
    save_file(file=T.numpy(), file_format='pkl', file_name='T', file_path=f'{project_filepath}/data/')

    model_f = TimeGAN_trainer(device=device, dataset=dataset, max_seq_len=seq_len, p=p, project_filepath=project_filepath)
    
    torch.save(model_f, f'{project_filepath}/models/model.pth')
    
    print("\nGenerating Data...")
    
    generated_f1 = timegan_generator(model=model_f, device=device, T=T, max_seq_len=seq_len, Z_dim=p, project_filepath=project_filepath)

    # gen_data_unpadded = unpad_arrays(generated_f1,T)
    # ori_data_unpadded = unpad_arrays(X, T)
    # window_length = 10
    # # the code below works with numpy arrays, convert X to numpy array
    # X = X.numpy()
    # for i, x_i in enumerate(gen_data_unpadded):
    #     for j in range(46):
    #         segments_ds = segment_time_series(ori_data_unpadded[i][:,j], window_length)
    #         segments_gen_ds = segment_time_series(x_i[:,j], window_length)
    #         for k in range(len(segments_ds)):
    #             std_segment = np.std(segments_ds[k])
    #             noise_segment = np.random.normal(scale=std_segment, size=len(segments_ds[k]))
    #             segments_gen_ds[k] += noise_segment
    
    save_file(file=generated_f1, file_format='pkl', file_name=f'generated', file_path=f'{project_filepath}/data/')
