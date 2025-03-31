import torch as th 
import numpy as np
from warnings import warn

class QsarDataset:
    def __init__(self, exp_trainset, qsar_trainset, exp_testset, qsar_testset, assay_names):
        '''
        Args:
            exp_trainset: numpy array of shape (#data points, #assays)
            qsar_trainset: numpy array of shape (#data points, #assays)
            exp_testset: numpy array of shape (#data points, #assays)
            qsar_testset: numpy array of shape (#data points, #assays)
            assay_names: numpy array of shape (#assays)
        '''
        self.exp_trainset = exp_trainset
        self.qsar_trainset = qsar_trainset
        self.exp_testset = exp_testset
        self.qsar_testset = qsar_testset

        self.feature_size = exp_trainset.shape[1]

        self.assay_names = assay_names

    @property
    def global_mean(self):
        return np.nanmean(self.exp_trainset, axis=0)
    
    @property
    def global_std(self):
        return np.nanstd(self.exp_trainset, axis=0)
    
    def check_gaussianity(self):
        deviation = self.exp_trainset - self.qsar_trainset
        task_warn = []
        for idx, assay_name in enumerate(self.assay_names):
            dev_notnan = deviation[:, idx][~np.isnan(deviation[:, idx])]
            dev_mean = np.mean(dev_notnan)
            dev_std = np.std(dev_notnan)

            dev_kurtosis = ((dev_notnan / dev_std)**4).mean() - 3
            if np.abs(dev_mean) > dev_std:
                warn("Deviation of QSAR from assay-{} data (idx={}): mean={}, std={}. Mean lies outside one std, suggesting bad gaussianity.".format(
                    assay_name, idx, dev_mean[idx], dev_std[idx]))
            if dev_kurtosis > 10:
                warn_str = "Deviation of QSAR from assay-{} data (idx={}): kurtosis={}. Kurtosis > 10 suggests bad gaussianity.".format(
                    assay_name, idx, dev_kurtosis)
                warn(warn_str)
                task_warn.append(assay_name)
            # outlier = np.abs(dev_notnan) > (2 * dev_std)
            # n_outlier = np.sum(outlier.astype(int))
            # print("task-{}, ".format(assay_name), 'ratio of outlier (>2*sigma)=', n_outlier / dev_notnan.shape[0])
        return task_warn
        
    def remove_outlier(self, task_warn, sigma=2):
        ## remove from self.exp_trainset, column by column, those data points whose deviation is larger than sigma*dev_std
        deviation = self.exp_trainset - self.qsar_trainset
        for idx, assay_name in enumerate(self.assay_names):
            if (assay_name in task_warn) is False:
                continue
            dev = deviation[:, idx]
            dev_mean = np.nanmean(dev)
            dev_std = np.nanstd(dev)
            filter_outlier = np.abs(dev - dev_mean) > (sigma * dev_std)
            print("remove outlier of assay-{} (idx={}): {} data points removed.".format(assay_name, idx, filter_outlier.sum()))
            self.exp_trainset[filter_outlier,idx] = np.nan
        return

    def get_torch_dataset(self, dataset_type, dtype=th.float32, device='cpu', normalize=True):
        '''
        Args:
            type: 'train' or 'test'
            dtype: torch data type
            device: torch device
            normalize: whether to normalize the data
        Returns:
            A TorchDataset
        '''
        if dataset_type == 'train':
            exp_data = self.exp_trainset
            qsar_data = self.qsar_trainset
        elif dataset_type == 'test':
            exp_data = self.exp_testset
            qsar_data = self.qsar_testset
        else:
            raise ValueError
        
        if normalize:
            global_mean = self.global_mean
            global_std = self.global_std
            exp_data = (exp_data - global_mean)/global_std
            qsar_data = (qsar_data - global_mean)/global_std
        
        exp_data = th.tensor(exp_data, dtype=dtype, device=device)
        qsar_data = th.tensor(qsar_data, dtype=dtype, device=device)
        return TorchDataset(
            exp_data = exp_data, 
            qsar_data = qsar_data,
            assay_names = self.assay_names)


class TorchDataset(th.utils.data.Dataset):
    def __init__(self, exp_data, qsar_data, assay_names):
        self.exp_data = exp_data
        self.qsar_data = qsar_data
        self.assay_names = assay_names
    
    def __len__(self):
        return len(self.exp_data)
    
    def __getitem__(self, idx):
        return {
            "exp_data": self.exp_data[idx],
            "qsar_data": self.qsar_data[idx]
        }
    