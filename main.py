import os
import sys
import numpy as np
import torch as th
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_squared_error
from data_tools import QsarDataset
from model import QComp
from train import trainer
from utilities import th2np, cross_count
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def pearsonr2_function(y_true, y_pred):
    return stats.pearsonr(y_true, y_pred)[0] ** 2

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
dtype = th.float32


################################################## 1. Setting ##################################################

idx = 0

fold_name = 'fold_' + str(idx)
n_threshold_mute = 10

#################### data path ####################
dataset_name = "public_data_results"
dataset_path = './{}/random_split_data_results/{}'.format(dataset_name, fold_name)

##### exp data #####
exp_trainset = pd.read_csv(os.path.join(dataset_path, 'public_admet_data_random_{}_train_set.csv'.format(fold_name)))
exp_testset = pd.read_csv(os.path.join(dataset_path, 'public_admet_data_random_{}_test_set.csv'.format(fold_name)))
exp_trainset = exp_trainset.drop(columns=["smiles"])
exp_testset = exp_testset.drop(columns=["smiles"])

##### qsar data #####
qsar_trainset = pd.read_csv(os.path.join(dataset_path, 'chemprop_multitask_pred/public_admet_data_random_{}_train_set_model_pred.csv'.format(fold_name)))
qsar_trainset = qsar_trainset[exp_trainset.columns]
qsar_testset = pd.read_csv(os.path.join(dataset_path, 'chemprop_multitask_pred/public_admet_data_random_{}_test_set_model_pred.csv'.format(fold_name)))
qsar_testset = qsar_testset[exp_testset.columns]



##### mute mask #####
count_matrix = cross_count(exp_trainset)
mute_mask = th.tensor(count_matrix > n_threshold_mute, dtype=dtype, device=device)

##### tasks #####
tasks = np.array(exp_trainset.columns)

##### to numpy #####
exp_trainset = exp_trainset.to_numpy()
exp_testset = exp_testset.to_numpy()
qsar_trainset = qsar_trainset.to_numpy()
qsar_testset = qsar_testset.to_numpy()


#################### training ####################
num_epoches = 15
batch_size = 100

flag_regularize = False

lr = 0.001
step_size = 1
gamma = 0.5

#################### output frequency ####################
freq_output_loss = 1
freq_output_figure = 500

#################### metrics ####################
metric_list = [
        pearsonr2_function, 
        r2_score,
        mean_squared_error, 
        ]
metric_name_list = [
                    "pearson_r2_score",
                    "r2_score",
                    "mean_squared_error",
                    ]


################################################## 2. Data Loading ##################################################
dataset = QsarDataset(exp_trainset, qsar_trainset, exp_testset, qsar_testset, assay_names=tasks)

trainset = dataset.get_torch_dataset('train', normalize=True)
testset = dataset.get_torch_dataset('test', normalize=True)

train_loader = th.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


################################################## 3. Model ##################################################
data_std = np.nanstd(trainset.exp_data, axis=0)
qcomp = QComp(size=dataset.feature_size, diagonal=data_std, mute_mask=mute_mask)


################################################## 4. Training ##################################################
trainer_obj = trainer(qcomp, lr, step_size, gamma)

current_time = datetime.now().strftime("%b%d_%H-%M-%S")

current_time += '_lr' + str(lr)

log_dir = os.path.join(dataset_path, "runs", current_time)
tb = SummaryWriter(log_dir=log_dir)


for idx_epoch in range(num_epoches):
    for idx,batch in enumerate(train_loader):

        n_iter = idx_epoch * len(train_loader) + idx

        loss = trainer_obj.train_step(batch["exp_data"], batch["qsar_data"], flag_regularize=flag_regularize)

        if n_iter % freq_output_figure == 0:
            with th.no_grad():

                for metric, metric_name in zip(metric_list, metric_name_list):
                    trainer_obj.impute_dataset(testset, metric, metric_name, tb, True, n_iter)

                th.save(qcomp.state_dict(), os.path.join(tb.log_dir, "model.pth"))

        if n_iter % freq_output_loss == 0:
            with th.no_grad():
                print("-------------------- n_iter: {} --------------------".format(n_iter))
                print("Epoch", idx_epoch, "Batch", idx, "Loss", loss.item(), "Learning Rate", trainer_obj.scheduler.get_lr()[0])
                tb.add_scalar("loss", loss.item(), n_iter)
                tb.add_scalar("learning_rate", trainer_obj.scheduler.get_lr()[0], n_iter)

                eigenv_sqrt = th.linalg.eigvalsh(qcomp.forward()) ** 0.5
                print("eigenv_sqrt of Sigma", eigenv_sqrt)
                tb.add_scalar("eigenv_sqrt max", th2np(eigenv_sqrt)[-1], n_iter)
                tb.add_scalar("eigenv_sqrt min", th2np(eigenv_sqrt)[0], n_iter)

                print("transform_matrix", th2np(qcomp.qsar_trans_M).diagonal())

                print("-----------------------------------------------------------------")

    trainer_obj.scheduler_step()
