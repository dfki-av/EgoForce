import numpy as np
from torch.utils.data import Dataset
from datasets import HOT3DLoader, ArcticLoader, H2OLoader, HandCOLoader, ReInterHandEgoLoader, HO3DV2Loader


class CombinedDataset(Dataset):
    def __init__(self, data_root, split, get_camera=False, **kwargs) -> None:
        self.config = kwargs['config']

        self.split = split
        self.get_camera = get_camera

        assert self.split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"

        if self.split == 'train':
            self.datasets = []

            self.datasets += [ReInterHandEgoLoader(self.config.DATASET.REINTERHAND_ROOT, split, get_camera, config=self.config)]

            self.datasets += [HandCOLoader(self.config.DATASET.HANDCO_ROOT, split, get_camera, cam=c, config=self.config) for c in [0, 1, 2, 3, 4, 5, 6, 7]]             

            self.datasets += [H2OLoader(self.config.DATASET.H2O_ROOT, split, get_camera, cam=c, config=self.config) for c in [2, 3, 4]]       

            self.datasets += [ArcticLoader(self.config.DATASET.ARCTIC_ROOT, split, get_camera, cam=c, config=self.config) for c in [
                                                                                                                0, 
                                                                                                                1, 
                                                                                                                2, 
                                                                                                                #3, 
                                                                                                                4, 
                                                                                                                5, 
                                                                                                                #6, 
                                                                                                                7, 
                                                                                                                8
                                                                                                                ]] 
            self.datasets += [HOT3DLoader(self.config.DATASET.HOT3D_ROOT, split, get_camera, cam=c, config=self.config) for c in [0]]
            self.datasets += [HO3DV2Loader(self.config.DATASET.HO3D_ROOT, split, get_camera, cam=c, config=self.config) for c in [0]] 
        else:
            if self.config.DATASET.NAME == 'ARCTIC':
                self.datasets = [ArcticLoader(self.config.DATASET.ARCTIC_ROOT, split, get_camera, config=self.config)]
            elif self.config.DATASET.NAME == 'H2O':
                self.datasets = [H2OLoader(self.config.DATASET.H2O_ROOT, split, get_camera, config=self.config)]
            elif self.config.DATASET.NAME == 'HOT3D':
                self.datasets = [HOT3DLoader(self.config.DATASET.HOT3D_ROOT, split, get_camera, config=self.config)]
            elif self.config.DATASET.NAME == 'HO3D':
                self.datasets = [HO3DV2Loader(self.config.DATASET.HO3D_ROOT, split, get_camera, config=self.config)]
            else:
                raise ValueError(f"Unknown dataset name: {self.config.DATASET.NAME}")

        self.lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(self.lengths)])
    
        self.n_samples = self.cumulative_lengths[-1]

        print(f"Combined dataset length: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        dataset_idx = int(np.searchsorted(self.cumulative_lengths, index, side='right')) - 1
        sample_idx = int(index - self.cumulative_lengths[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]
        


class ArcticCombinedLoader(Dataset):
    def __init__(self, data_root, split, get_camera=False, **kwargs) -> None:
        self.config = kwargs['config']

        self.split = split
        self.get_camera = get_camera

        assert self.split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"

        if self.split == 'train':
            self.datasets = [ArcticLoader(self.config.DATASET.ARCTIC_ROOT, split, get_camera, cam=c, config=self.config) for c in [
                                                                                                                0, 
                                                                                                                1, 
                                                                                                                2, 
                                                                                                                3, 
                                                                                                                4, 
                                                                                                                5, 
                                                                                                                6, 
                                                                                                                7, 
                                                                                                                8
                                                                                                                ]] 
        else:
            self.datasets = [ArcticLoader(self.config.DATASET.ARCTIC_ROOT, split, get_camera, config=self.config)]

        self.lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(self.lengths)])
    
        self.n_samples = self.cumulative_lengths[-1]

        print(f"ArcticCombinedLoader length: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        dataset_idx = int(np.searchsorted(self.cumulative_lengths, index, side='right')) - 1
        sample_idx = int(index - self.cumulative_lengths[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]



class H2OCombinedLoader(Dataset):
    def __init__(self, data_root, split, get_camera=False, **kwargs) -> None:
        self.config = kwargs['config']

        self.split = split
        self.get_camera = get_camera

        assert self.split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"

        if self.split == 'train':
            self.datasets = [H2OLoader(self.config.DATASET.H2O_ROOT, split, get_camera, cam=c, config=self.config) for c in [2, 3, 4]]        
        else:
            self.datasets = [H2OLoader(self.config.DATASET.H2O_ROOT, split, get_camera, config=self.config)]

        self.lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(self.lengths)])
    
        self.n_samples = self.cumulative_lengths[-1]

        print(f"H2OCombinedLoader length: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        dataset_idx = int(np.searchsorted(self.cumulative_lengths, index, side='right')) - 1
        sample_idx = int(index - self.cumulative_lengths[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]


class ArcticExoLoader(Dataset):
    def __init__(self, data_root, split, get_camera=False, **kwargs) -> None:
        self.config = kwargs['config']

        self.split = split
        self.get_camera = get_camera

        assert self.split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"

        if self.split == 'train':
            self.datasets = [ArcticLoader(self.config.DATASET.ARCTIC_ROOT, split, get_camera, cam=c, config=self.config) for c in [
                                                                                                                1, 
                                                                                                                2, 
                                                                                                                3, 
                                                                                                                4, 
                                                                                                                5, 
                                                                                                                6, 
                                                                                                                7, 
                                                                                                                8
                                                                                                                ]] 
        else:
            self.datasets = [ArcticLoader(self.config.DATASET.ARCTIC_ROOT, split, get_camera, config=self.config)]

        self.lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(self.lengths)])
    
        self.n_samples = self.cumulative_lengths[-1]

        print(f"ArcticExoLoader length: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        dataset_idx = int(np.searchsorted(self.cumulative_lengths, index, side='right')) - 1
        sample_idx = int(index - self.cumulative_lengths[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]



class H2OExoLoader(Dataset):
    def __init__(self, data_root, split, get_camera=False, **kwargs) -> None:
        self.config = kwargs['config']

        self.split = split
        self.get_camera = get_camera

        assert self.split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"

        if self.split == 'train':
            self.datasets = [H2OLoader(self.config.DATASET.H2O_ROOT, split, get_camera, cam=c, config=self.config) for c in [2, 3]]        
        else:
            self.datasets = [H2OLoader(self.config.DATASET.H2O_ROOT, split, get_camera, config=self.config)]

        self.lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(self.lengths)])
    
        self.n_samples = self.cumulative_lengths[-1]

        print(f"H2OExoLoader length: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        dataset_idx = int(np.searchsorted(self.cumulative_lengths, index, side='right')) - 1
        sample_idx = int(index - self.cumulative_lengths[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]
