import wandb

from .wmtask import *

class WMTaskEq:
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def jac(self, hiddens, t=None, discrete=False):
        return compute_model_jacs(self.model, hiddens, self.params['dt'], self.params['tau'], discrete=discrete)
    
    def rhs(self, hiddens, t=None):
        return compute_model_rhs(self.model, hiddens, self.params['dt'], self.params['tau'])

def load_wmtask_model(project, name, model_to_load='final'):
    api = wandb.Api()
    runs = api.runs(project)

    run = [run for run in runs if run.name == name][0]
    raise ValueError("No save directory for WMTaskModels found due to anonymizing code.")
    model_load_dir = os.path.join(run.config['save_dir'], project, name)

    params = run.config
    params = OmegaConf.create(params)

    if 'enforce_fixation' not in params:
        params['enforce_fixation'] = False

    # models_to_load = ['init', 1, params['max_epochs'] - 1]
    # models_to_load = ['init', params['max_epochs'] - 1]
    if model_to_load == 'final':
        model_to_load = params['max_epochs'] - 1
    elif model_to_load == 'init':
        pass
    elif isinstance(model_to_load, int):
        model_to_load = model_to_load
    else:
        raise ValueError(f"model_to_load must be 'final', 'init', or an integer, got {model_to_load}")
    # LOAD MODEL


    if model_to_load == 'init':
        torch.manual_seed(params['random_state'])
        model = BiologicalRNN(params['input_dim'], params['hidden_dim'], output_dim=params['num_stimuli'], dt=params['dt'], tau=params['tau'], enforce_fixation=params['enforce_fixation'])
        # model_names.append('init')
    else:
        filename = f"model-epoch={model_to_load}.ckpt"
        # model_names.append(filename)
        if torch.cuda.is_available():  
            state_dict = torch.load(os.path.join(model_load_dir, filename), weights_only=False)['state_dict']
        else:
            state_dict = torch.load(os.path.join(model_load_dir, filename), weights_only=False, map_location='cpu')['state_dict']
        state_dict = {k.split('.')[1]: v for k, v in state_dict.items()}
        if 'enforce_fixation' in params.keys():
            model = BiologicalRNN(params['input_dim'], params['hidden_dim'], output_dim=params['num_stimuli'], dt=params['dt'], tau=params['tau'], enforce_fixation=params['enforce_fixation'])
        else:
            model = BiologicalRNN(params['input_dim'], params['hidden_dim'], output_dim=params['num_stimuli'], dt=params['dt'], tau=params['tau'])
        model.load_state_dict(state_dict)
        print(f"loaded wmtask RNN model checkpoint {model_to_load}")
    
    return model, params, 

def generate_wmtask_data(params):
    np.random.seed(params.random_state)
    torch.manual_seed(params.random_state)
    color_stimuli = nn.functional.one_hot(torch.arange(params.num_stimuli), params.num_stimuli).type(torch.FloatTensor)

    color_nums = torch.arange(4)
    color1_index = torch.randint(low=0, high=params.num_stimuli, size=(params.num_trials,))
    color1_input = color_stimuli[color1_index]
    color2_index = torch.tensor([torch.cat((color_nums[:c_ind], color_nums[c_ind + 1:]))[torch.randint(low=0, high=3, size=(1,))][0] for c_ind in color1_index])
    color2_input = color_stimuli[color2_index]

    context_input = nn.functional.one_hot(torch.randint(low=0, high=2, size=(params.num_trials,)), 2)
    color_labels = torch.cat((color1_index.unsqueeze(-1), color2_index.unsqueeze(-1)), axis=1)[context_input.type(torch.BoolTensor)]

    stacked_inputs = torch.cat((color1_input, color2_input, context_input), axis=1)

    train_inds = np.sort(np.random.choice(np.arange(params.num_trials), size=(int(params.train_percent*params.num_trials)), replace=False))
    val_inds = np.array([i for i in np.arange(params.num_trials) if i not in train_inds])
    test_inds = np.sort(np.random.choice(val_inds, size=(150,)))
    if 'enforce_fixation' in params.keys():
        all_dataset = WMSelectionDataset(stacked_inputs, color_labels, params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
        train_dataset = WMSelectionDataset(stacked_inputs[train_inds], color_labels[train_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
        val_dataset = WMSelectionDataset(stacked_inputs[val_inds], color_labels[val_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
        test_dataset = WMSelectionDataset(stacked_inputs[test_inds], color_labels[test_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
    else:
        all_dataset = WMSelectionDataset(stacked_inputs, color_labels, params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)
        train_dataset = WMSelectionDataset(stacked_inputs[train_inds], color_labels[train_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)
        val_dataset = WMSelectionDataset(stacked_inputs[val_inds], color_labels[val_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)
        test_dataset = WMSelectionDataset(stacked_inputs[test_inds], color_labels[test_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)

    num_workers = 1
    all_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)

    return all_dataloader, train_dataloader, val_dataloader, test_dataloader

def generate_model_trajectories(model, dataloader, params, verbose=False):
    areas = ['V4', 'PFC']
    area_inds = [np.arange(params['N1']), params['N1'] + np.arange(params['N2'])]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hiddens = get_hiddens(model, dataloader, verbose=verbose)

    return hiddens