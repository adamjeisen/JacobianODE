import hydra
import lightning as L
import logging
import numpy as np
from omegaconf import OmegaConf
import os
from sklearn.model_selection import KFold
import torch
from torch import nn
from tqdm.auto import tqdm
import wandb

from ..control.controllability import compute_folded_accuracy

# ---------------
# Data analysis
# ---------------

def get_hiddens(model, dataloader_to_use, device='cpu', verbose=False):
    hiddens_all = torch.zeros(dataloader_to_use.dataset.labels.shape[0], dataloader_to_use.dataset.total_t, model.hidden_dim)
    model = model.to(device)
    with torch.no_grad():
        batch_loc = 0
        for input_seq, labels in tqdm(dataloader_to_use, disable=not verbose):
            hiddens = torch.zeros(input_seq.shape[0], input_seq.shape[1], model.hidden_dim)
            for i in range(input_seq.shape[1]):
                if i == 0:
                    out, hidden = model(input_seq[:, [i]].to(device))
                else:
                    out, hidden = model(input_seq[:, [i]].to(device), hidden)
                hiddens[:, i] = hidden.cpu()
            hiddens_all[batch_loc:batch_loc + input_seq.shape[0]] = hiddens

            batch_loc += input_seq.shape[0]

    return hiddens_all

def ELU_deriv(h):
    deriv = torch.zeros(h.shape).type(h.dtype).to(h.device)
    deriv[h > 0] = 1
    deriv[h <= 0] = torch.exp(h[h <= 0])
    return deriv

def compute_model_jacs(model, h, dt, tau, discrete=False):
    if discrete:
        Js = torch.eye(h.shape[-1]).unsqueeze(0).type(h.dtype).to(h.device) + (dt/tau)*(-torch.eye(h.shape[-1]).unsqueeze(0).type(h.dtype).to(h.device) + model.W_hh.detach().type(h.dtype).to(h.device) @ torch.diag_embed(ELU_deriv(h).type(h.dtype).to(h.device)))
    else:
        Js = (1/tau)*(-torch.eye(h.shape[-1]).unsqueeze(0).type(h.dtype).to(h.device) + model.W_hh.detach().type(h.dtype).to(h.device) @ torch.diag_embed(ELU_deriv(h).type(h.dtype).to(h.device)))
    return Js

def compute_model_rhs(model, h, dt, tau):
    return (1/tau)*(-h + (model.W_hh.detach().type(h.dtype).to(h.device) @ nn.ELU()(h.unsqueeze(-1))).squeeze(-1).type(h.dtype).to(h.device))

def get_all_test_accuracies(model, area_hiddens, com_signals, areas, area_inds, dataloader_to_use, com_subspace_signals=None, n_splits=5, variable='choice'):
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    test_accuracies = {}

    if variable == 'choice':
        labels = dataloader_to_use.dataset.labels
    elif variable == 'top':
        labels = dataloader_to_use.dataset.inputs[:, :model.input_dim].argmax(dim=-1)
    elif variable == 'bottom':
        labels = dataloader_to_use.dataset.inputs[:, model.input_dim:2*model.input_dim].argmax(dim=-1)
    
    n_splits = 5
    split_inds = []
    for train_inds, test_inds in KFold(n_splits=n_splits, shuffle=True).split(labels):
        split_inds.append((train_inds, test_inds))
    
    iterator = tqdm(total=n_splits*(dataloader_to_use.dataset.total_t - 1)*len(areas)*(8 if com_subspace_signals is None else 11))
    
    subspace_dim = None
    # subspace_dim = 10
    # subspace_dim = np.where(np.cumsum(com_SVD['V4']['PFC']['S'].mean(axis=(0, 1))**2)/np.sum((com_SVD['V4']['PFC']['S'].mean(axis=(0, 1))**2).numpy()) > 0.75)[0][0]
    
    for area1, inds1 in zip(areas, area_inds):
        test_accuracies[area1] = {}
        for area2, inds2 in zip(areas, area_inds):
            if area1 != area2:
                
                test_accuracies[area1][area2] = {}
                # -----------------
                # SENDER
                # -----------------
                if subspace_dim is None:
                    sdim = len(inds2)
                else:
                    sdim = subspace_dim
                
                dx = area_hiddens[area2][:, 1:] - area_hiddens[area2][:, :-1]
                com_signal = com_signals[area1][area2]['sender']
                
    
                test_accuracies[area1][area2]['com_send'] = compute_folded_accuracy(com_signal[:, :, :], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                test_accuracies[area1][area2]['dx'] = compute_folded_accuracy(dx, labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                test_accuracies[area1][area2]['x'] = compute_folded_accuracy(area_hiddens[area2][:, 1:], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                test_accuracies[area1][area2]['phi_x'] = compute_folded_accuracy(model.activation(area_hiddens[area2][:, 1:]), labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                if com_subspace_signals is not None:
                    com_sub_signal = com_subspace_signals[area1][area2]['sender']
                    test_accuracies[area1][area2]['com_sub_send'] = compute_folded_accuracy(com_sub_signal[:, 1:, :], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                    com_sub_signal = com_subspace_signals[area1][area2]['overall']
                    test_accuracies[area1][area2]['com_sub_overall'] = compute_folded_accuracy(com_sub_signal[:, 1:, :], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                    com_sub_signal = com_subspace_signals[area1][area2]['overall_v2']
                    test_accuracies[area1][area2]['com_sub_overall_v2'] = compute_folded_accuracy(com_sub_signal[:, 1:, :], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
    
                with torch.no_grad():
                    # _, NJ_S, NJ_Vh = torch.linalg.svd(model.W_hh.cpu()[inds1][:, inds2])
                    # NJ_dirs = (torch.diag_embed(NJ_S) @ NJ_Vh)[:sdim]
                    NJ_dirs = model.W_hh.cpu()[inds1][:, inds2]
                    nj_signals = (NJ_dirs @ model.activation(area_hiddens[area2][:, 1:]).unsqueeze(-1)).squeeze(-1)
                    test_accuracies[area1][area2]['nj'] = compute_folded_accuracy(nj_signals, labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                
                # -----------------
                # RECEIVER
                # -----------------
                if subspace_dim is None:
                    sdim = len(inds1)
                else:
                    sdim = subspace_dim
                
                dy = area_hiddens[area1][:, 1:] - area_hiddens[area1][:, :-1]
                com_signal = com_signals[area1][area2]['receiver']
    
                test_accuracies[area1][area2]['com_recv'] = compute_folded_accuracy(com_signal[:, :, :], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                test_accuracies[area1][area2]['dy'] = compute_folded_accuracy(dy, labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
                test_accuracies[area1][area2]['y'] = compute_folded_accuracy(area_hiddens[area1][:, 1:], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
    
                if com_subspace_signals is not None:
                    com_sub_signal = com_subspace_signals[area1][area2]['receiver']
                    test_accuracies[area1][area2]['com_sub_recv'] = compute_folded_accuracy(com_sub_signal[:, 1:, :], labels, split_inds, iterator, verbose=True, n_splits=n_splits)[1]
    
    iterator.close()

    return test_accuracies

# ----------------
# Data generation
# ----------------

# Custom Dataset class
class WMSelectionDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, dt, input_dim, fixation_time, stimuli_time, delay1_time, cue_time, delay2_time, response_time, enforce_fixation):
        self.inputs = inputs
        self.labels = labels

        self.input_dim = input_dim
        
        self.fixation_time = fixation_time
        self.stimuli_time = stimuli_time
        self.delay1_time = delay1_time
        self.cue_time = cue_time
        self.delay2_time = delay2_time
        response_time = response_time if response_time is not None else dt
        self.response_time = response_time
        self.n_response_t = int(np.round(response_time/dt))
        self.total_t = int(np.round((fixation_time + stimuli_time + delay1_time + cue_time + delay2_time + response_time)/dt))
        self.stim_start_t = int(np.round((self.fixation_time/dt)))
        self.stim_end_t = int(np.round((self.fixation_time + self.stimuli_time)/dt))
        self.cue_start_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time)/dt))
        self.cue_end_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time + self.cue_time)/dt))
        self.response_start_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time + self.cue_time + self.delay2_time)/dt))
        self.response_end_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time + self.cue_time + self.delay2_time + self.response_time)/dt))

        self.enforce_fixation = enforce_fixation

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        stacked_input = self.inputs[idx]
        
        input_sample = torch.zeros(self.total_t, len(stacked_input) + (1 if self.enforce_fixation else 0)).type(stacked_input.dtype).to(stacked_input.device)
        input_sample[self.stim_start_t:self.stim_end_t, :self.input_dim*2] = stacked_input[:self.input_dim*2] # stimulus inputs
        input_sample[self.cue_start_t:self.cue_end_t, self.input_dim*2:self.input_dim*2 + 2] = stacked_input[self.input_dim*2:] # cue input
        if self.enforce_fixation:
            input_sample[:self.response_start_t, -1] = 1
        
        # input_sample = stacked_input)
        label_sample = self.labels[idx].repeat(self.n_response_t)
        return input_sample, label_sample

# -------------
# Biological RNN
# -------------

def make_matrix(d, eig_lower_bound=1e-1):
    eigvals_real = - torch.rand(int(d/2))*(1e-1)
    freqs = torch.rand(int(d/2))*2*np.pi
    eigvals_diag = torch.zeros(d, d)
    for i in range(0, d, 2):
        eigvals_diag[i, i] = eigvals_real[int(i/2)]
        eigvals_diag[i + 1, i + 1] = eigvals_real[int(i/2)]
        eigvals_diag[i, i+1] = -freqs[int(i/2)]
        eigvals_diag[i+1, i] = freqs[int(i/2)]
    eigvecs = torch.linalg.qr(torch.randn(d, d))[0]
    M = torch.matrix_exp(eigvecs @ eigvals_diag @ torch.linalg.pinv(eigvecs))

    return M

class BiologicalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dt, tau, bias=True, eig_lower_bound=1e-1, enforce_fixation=False):
        super(BiologicalRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        output_dim = output_dim + (2 if enforce_fixation else 0)
        self.output_dim = output_dim
        self.dt = dt
        self.tau = tau
        self.eig_lower_bound = eig_lower_bound
        # self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        self.activation = nn.ELU()

        W_hi = torch.randn(hidden_dim, input_dim + input_dim + 2 + (1 if enforce_fixation else 0))
        W_hi /= torch.linalg.norm(W_hi, ord=2, dim=(-2, -1))
        self.W_hi = nn.Parameter(W_hi)
        
        M = torch.zeros(hidden_dim, hidden_dim)
        M[:int(hidden_dim/2), :int(hidden_dim/2)] = make_matrix(int(hidden_dim/2), eig_lower_bound=eig_lower_bound)
        M[int(hidden_dim/2):, int(hidden_dim/2):] = make_matrix(int(hidden_dim/2), eig_lower_bound=eig_lower_bound)

        W_inter1 = torch.randn(int(hidden_dim/2), int(hidden_dim/2))
        W_inter1 /= torch.linalg.norm(W_inter1, ord=2, dim=(-2, -1))
        W_inter1 *= 0.05
        M[:int(hidden_dim/2), int(hidden_dim/2):] = W_inter1
        W_inter2 = torch.randn(int(hidden_dim/2), int(hidden_dim/2))
        W_inter2 /= torch.linalg.norm(W_inter2, ord=2, dim=(-2, -1))
        W_inter2 *= 0.05
        M[int(hidden_dim/2):, :int(hidden_dim/2)] = W_inter2

        self.W_hh = nn.Parameter(M)
        
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        self.input_mask = torch.zeros(self.W_hi.shape)
        self.input_mask[:int(hidden_dim/2)] = 1

        W_oh = torch.randn(output_dim, hidden_dim)
        W_oh /= torch.linalg.norm(W_oh, ord=2, dim=(-2, -1))
        self.W_oh = nn.Parameter(W_oh)

        self.output_mask = torch.zeros(self.W_oh.shape)
        self.output_mask[:, int(hidden_dim/2):] = 1

        self.hidden_init = nn.Parameter(torch.randn(self.hidden_dim))

    def forward(self, x, hidden=None):
        """
        Given an input sequence x from time 0:t
        """
        h0 = hidden
        if h0 is None:
            if len(x.shape) == 3:
                h0 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
                h0[:] = self.hidden_init
            else:
                h0 = torch.zeros(self.hidden_dim).to(x.device)
                h0 = self.hidden_init
                
        h = h0
        
        squeeze = 0
        if len(x.shape) == 2:
            squeeze = 1
            x = x.unsqueeze(0)
        elif len(x.shape) == 1:
            squeeze = 2
            x = x.unsqueeze(0).unsqueeze(0)

        self.input_mask = self.input_mask.type(x.dtype).to(x.device)
        self.output_mask = self.output_mask.type(x.dtype).to(x.device)

        outs = []
        for i in range(x.size(1)):

            # h = h + (dt/tau)*(-h + self.W_hi(x[:, i]) + self.W_hh(self.activation(h)))
            h = h + (self.dt/self.tau)*(-h + ((self.W_hi * self.input_mask) @ x[:, i].unsqueeze(-1)).squeeze(-1) + (self.W_hh @ self.activation(h).unsqueeze(-1)).squeeze(-1) + self.b)
            # h = h + (dt/tau)*(-h + self.activation(((self.W_hi * self.input_mask) @ x[:, i].unsqueeze(-1)).squeeze(-1) + (self.W_hh @ h.unsqueeze(-1)).squeeze(-1) + self.b))
            outs.append(((self.W_oh * self.output_mask) @ h.unsqueeze(-1)).squeeze(-1))
        outs = torch.stack(outs, dim=1)
        if squeeze == 1:
            outs = outs.squeeze(0)
        elif squeeze == 2:
            outs = outs.squeeze(0).squeeze(0)

        # outs, _ = self.RNN(x, h)
        # outs = self.W_oh(outs) 
    
        return outs, h

class LitBiologicalRNN(L.LightningModule):
    def __init__(self, model, save_dir=None, learning_rate=1e-4, enforce_fixation=False):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.save_dir = save_dir
    
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.enforce_fixation = enforce_fixation
    
    def model_step(self, batch, batch_idx, dataloader_idx=0, all_metrics=False, generate=None):

        input_seq, labels = batch
        out, hidden = self.model(input_seq)
        loss = self.criterion(out[:, -labels.shape[-1]:, :self.model.input_dim].transpose(-2, -1), labels)
        if self.enforce_fixation:
            fix_loss = self.criterion(out[:, :, self.model.input_dim:].transpose(-2, -1), input_seq[:, :, -1].type(torch.LongTensor).to(out.device))
            loss += fix_loss
            fix_accuracy = torch.sum(nn.Softmax(dim=-1)(out[:, :, self.model.input_dim:]).argmax(dim=-1) == input_seq[:, :, -1])/(input_seq.shape[0]*input_seq.shape[1])
        accuracy = torch.sum(nn.Softmax(dim=-1)(out[:, -labels.shape[-1]:, :self.model.input_dim]).argmax(dim=-1) == labels)/(labels.shape[0]*labels.shape[1])
        
        # return {'accuracy': accuracy, 'loss': loss}
        if self.enforce_fixation:
            return {'accuracy': accuracy, 'loss': loss, 'fix_accuracy': fix_accuracy}
        else:
            return {'accuracy': accuracy, 'loss': loss}

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        ret = self.model_step(batch, batch_idx)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", ret['loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_accuracy", ret['accuracy'], on_step=True, on_epoch=True, sync_dist=True)

        if self.enforce_fixation:
            self.log("train_fix_accuracy", ret['fix_accuracy'], on_step=True, on_epoch=True, sync_dist=True)
        
        return ret['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ret = self.model_step(batch, batch_idx)
        
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", ret['loss'], sync_dist=True)
        self.log("val_accuracy", ret['accuracy'], sync_dist=True)

        if self.enforce_fixation:
            self.log("val_fix_accuracy", ret['fix_accuracy'], sync_dist=True)

        return ret['loss']

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # one step predictions
        loss = self.model_step(batch, batch_idx)
        
        # plot_predictions(batch.cpu(), outputs.cpu(), outputs_gen.cpu(), metric_vals, metric_vals_gen, save_dir=self.save_dir)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# ----------------
# Running task code
# ----------------

log = logging.getLogger('WMTask Logger')

@hydra.main(config_path="conf", config_name="config.yaml", version_base='1.3')
def run_wmtask(cfg):
    log.info(f"dt = {cfg.wmtask_params.dt}")
    # ----------------
    # Update config
    # ----------------
    cfg.wmtask_params.input_dim = cfg.wmtask_params.num_stimuli
    cfg.wmtask_params.N2 = cfg.wmtask_params.N1
    cfg.wmtask_params.hidden_dim = cfg.wmtask_params.N1 + cfg.wmtask_params.N2

    # project_keys = ['fixation_time', 'stimuli_time', 'delay1_time', 'cue_time', 'delay2_time', 'num_stimuli', 'num_trials']
    # project = "WMSelectionTask__" + "__".join([f"{k}_{v}" for k, v, in cfg.wmtask_params.items() if k in project_keys])
    project = "__".join(["WMSelectionTask"] + [f"{k}_{v}" for k, v, in cfg.wmtask_params.items() if k in ['cue_time', 'response_time', 'enforce_fixation']])

    name_keys = ['N1', 'N2', 'tau', 'dt', 'eig_lower_bound', 'learning_rate', 'max_epochs', 'cue_time']
    name = "BiologicalRNN__" + "__".join([f"{k}_{v}" for k, v, in cfg.wmtask_params.items() if k in name_keys])

    model_save_dir = os.path.join(cfg.wmtask_params.save_dir, project, name)
    lit_dir = os.path.join(cfg.wmtask_params.save_dir, project, 'lightning')

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(lit_dir, exist_ok=True)

    log.info("Checking for preexisting runs...")
    api = wandb.Api()
    runs = api.runs(project)
    try:
        found_run = False
        for run in runs:
            if run.name == name:
                found_run = True
                break
        if found_run:
            print(f"Found run {name}, skipping...")
            log.info(f"Found run {name}, skipping...")
            return
    except ValueError:
        log.info(f"Project {project} does not exist!")

    # ----------------
    # Data generation
    # ----------------
    log.info("Generating data...")
    np.random.seed(cfg.wmtask_params.random_state)
    color_stimuli = nn.functional.one_hot(torch.arange(cfg.wmtask_params.num_stimuli), cfg.wmtask_params.num_stimuli).type(torch.FloatTensor)

    color_nums = torch.arange(4)
    color1_index = torch.randint(low=0, high=cfg.wmtask_params.num_stimuli, size=(cfg.wmtask_params.num_trials,))
    color1_input = color_stimuli[color1_index]
    color2_index = torch.tensor([torch.cat((color_nums[:c_ind], color_nums[c_ind + 1:]))[torch.randint(low=0, high=3, size=(1,))][0] for c_ind in color1_index])
    color2_input = color_stimuli[color2_index]

    context_input = nn.functional.one_hot(torch.randint(low=0, high=2, size=(cfg.wmtask_params.num_trials,)), 2)
    color_labels = torch.cat((color1_index.unsqueeze(-1), color2_index.unsqueeze(-1)), axis=1)[context_input.type(torch.BoolTensor)]

    stacked_inputs = torch.cat((color1_input, color2_input, context_input), axis=1)

    train_inds = np.sort(np.random.choice(np.arange(cfg.wmtask_params.num_trials), size=(int(cfg.wmtask_params.train_percent*cfg.wmtask_params.num_trials)), replace=False))
    val_inds = np.array([i for i in np.arange(cfg.wmtask_params.num_trials) if i not in train_inds])
    # train_dataset = WMSelectionDataset(stacked_inputs[train_inds], color_labels[train_inds], cfg.wmtask_params.dt, cfg.wmtask_params.input_dim, cfg.wmtask_params.fixation_time, cfg.wmtask_params.stimuli_time, cfg.wmtask_params.delay1_time, cfg.wmtask_params.cue_time, cfg.wmtask_params.delay2_time)
    # val_dataset = WMSelectionDataset(stacked_inputs[val_inds], color_labels[val_inds], cfg.wmtask_params.dt, cfg.wmtask_params.input_dim, cfg.wmtask_params.fixation_time, cfg.wmtask_params.stimuli_time, cfg.wmtask_params.delay1_time, cfg.wmtask_params.cue_time, cfg.wmtask_params.delay2_time)
    train_dataset = WMSelectionDataset(stacked_inputs[train_inds], color_labels[train_inds], cfg.wmtask_params.dt, cfg.wmtask_params.input_dim, cfg.wmtask_params.fixation_time, cfg.wmtask_params.stimuli_time, cfg.wmtask_params.delay1_time, cfg.wmtask_params.cue_time, cfg.wmtask_params.delay2_time, cfg.wmtask_params.response_time, cfg.wmtask_params.enforce_fixation)
    val_dataset = WMSelectionDataset(stacked_inputs[val_inds], color_labels[val_inds], cfg.wmtask_params.dt, cfg.wmtask_params.input_dim, cfg.wmtask_params.fixation_time, cfg.wmtask_params.stimuli_time, cfg.wmtask_params.delay1_time, cfg.wmtask_params.cue_time, cfg.wmtask_params.delay2_time, cfg.wmtask_params.response_time, cfg.wmtask_params.enforce_fixation)

    num_workers = 1
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.wmtask_params.batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.wmtask_params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)

    # ----------------
    # Lightning model
    # ----------------
    torch.manual_seed(cfg.wmtask_params.random_state)
    model = BiologicalRNN(cfg.wmtask_params.input_dim, cfg.wmtask_params.hidden_dim, output_dim=cfg.wmtask_params.num_stimuli, dt=cfg.wmtask_params.dt, tau=cfg.wmtask_params.tau, eig_lower_bound=cfg.wmtask_params.eig_lower_bound, enforce_fixation=cfg.wmtask_params.enforce_fixation)
    # model = LSTM(input_dim*2 + 2, hidden_dim, num_layers=6, output_dim=num_stimuli)
    lit_model = LitBiologicalRNN(model, learning_rate=cfg.wmtask_params.learning_rate, enforce_fixation=cfg.wmtask_params.enforce_fixation)
    logger = L.pytorch.loggers.WandbLogger(save_dir=lit_dir, log_model=True, name=name, project=project)
    logger.experiment.config.update(OmegaConf.to_container(cfg.wmtask_params))

    # Define the ModelCheckpoint callback to save a checkpoint at every epoch
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=None,          # Do not monitor any specific quantity
        dirpath=model_save_dir,  # Directory to save the checkpoints
        filename='model-{epoch}',  # Name format for saved checkpoints (only including epoch number)
        save_top_k = -1,
        every_n_epochs = 1,      # Save a checkpoint at every epoch
    )

    torch.autograd.set_detect_anomaly(True)
    trainer = L.Trainer(logger=logger, max_epochs=cfg.wmtask_params.max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    wandb.finish()

if __name__ == "__main__":
    run_wmtask()