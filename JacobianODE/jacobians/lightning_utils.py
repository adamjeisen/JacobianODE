from omegaconf import OmegaConf

def make_run_info(cfg):
    if cfg.data.data_type == 'dysts':
        data_cls = cfg.data.flow._target_.split('.')[-1]
    elif cfg.data.data_type == 'wmtask':
        data_cls = 'WMTask'

    name = tuple([f"{key}_{value}" for key, value in cfg.model.params.items() if value is not None and key not in ['_target_', '_partial_', 'embedder_kwargs', 'input_dim', 'output_dim']])
    if 'deriv_params' in cfg.model and cfg.model.deriv_params is not None:
        name = name + tuple([f"{key}_{value}" for key, value in cfg.model.deriv_params.items() if value is not None and key not in ['_target_', '_partial_', 'embedder_kwargs', 'input_dim', 'output_dim']])
    name = name + tuple([f"{key}_{value}" for key, value in cfg.training.items() if key in ['batch_size', 'save_top_k']])
    name = name + tuple([f"{key}_{value:.4f}" if key in ['obs_noise_scale', 'obs_noise_scale_validation'] else f"{key}_{value}" for key, value in cfg.training.lightning.items() if value is not None and key not in ['_target_', 'eq']])
    name = (cfg.model.params._target_.split('.')[-1],) + name
    name = name + tuple([f"{key}_{value:.4f}" if key in ['obs_noise'] else f"{key}_{value}" for key, value in cfg.data.postprocessing.items()])
    name = name + tuple([f"{key}_{value}" for key, value in cfg.data.train_test_params.items() if key in ("n_delays")])
    name = name + tuple([f"{key}_{value}" for key, value in cfg.data.train_test_params.items() if key in ("seq_length")])
    name = name + tuple([f"{key}_{value}" for key, value in cfg.data.trajectory_params.items() if key in ("n_periods", "pts_per_period", "standardize", "noise", "num_ics")])
    name = "__".join(name)

    project = (data_cls,'JacobianEstimation')
    project = "__".join(project)

    return name, project

def reverse_wandb_config(config):
    data_type = config['data_type'] if 'data_type' in config else 'dysts'
    flow_params = dict(
        random_state=config['random_state'],
    )
    if data_type == 'dysts':
        flow_params['_target_'] = 'CommunicationJacobians.dysts_sim.flows.' + config['data_cls']
        flow_params['dt'] = config['dt']
        trajectory_params = dict(
            n_periods=config['n_periods'],
            method=config['method'],
            resample=config['resample'],
            pts_per_period=config['pts_per_period'],
            return_times=config['return_times'],
            standardize=config['standardize'],
            noise=config['noise'],
            num_ics=config['num_ics'],
            traj_offset_sd=config['traj_offset_sd'],
            verbose=config['verbose'],
        )
    elif data_type == 'wmtask':
        flow_params['project'] = config['project']
        flow_params['name'] = config['name']
        trajectory_params = dict(
            dataloader_to_use=config['dataloader_to_use'],
            traj_window=config['traj_window'],
            model_to_load=config['model_to_load'] if 'model_to_load' in config else 'final',
        )

    postprocessing = dict(
        obs_noise=config['obs_noise'] if 'obs_noise' in config else False,
        filter_data=config['filter_data'] if 'filter_data' in config else False,
        low_pass=config['low_pass'] if 'low_pass' in config else None,
        high_pass=config['high_pass'] if 'high_pass' in config else None,
    )

    random_points_params = dict(
        n_points=config['n_points'] if 'n_points' in config else 50,
        n_steps=config['n_steps'] if 'n_steps' in config else 1,
    )
    random_points_params['n_validation_steps'] = config['n_validation_steps'] if 'n_validation_steps' in config else random_points_params['n_steps']


    random_points_reverse_params = dict(
        n_points=config['n_points'] if 'n_points' in config else random_points_params['n_points'],
        n_steps=config['n_steps'] if 'n_steps' in config else random_points_params['n_steps'],
        n_validation_steps=config['n_validation_steps'] if 'n_validation_steps' in config else random_points_params['n_steps'],
    )

    train_test_params = dict(
        seq_length=config['seq_length'],
        seq_spacing=config['seq_spacing'],
        train_percent=config['train_percent'],
        test_percent=config['test_percent'],
        split_by=config['split_by'],
        dtype=config['dtype'],
        verbose=config['verbose'],
    )
    delay_embedding_params = dict(
        n_delays=config['n_delays'] if 'n_delays' in config else 1,
        delay_spacing=config['delay_spacing'] if 'delay_spacing' in config else 1,
        observed_indices=config['observed_indices'] if 'observed_indices' in config else 'all',
    )
    train_test_params['delay_embedding_params'] = delay_embedding_params

    train_test_params['reverse_seq_length'] = config['reverse_seq_length'] if 'reverse_seq_length' in config else train_test_params['seq_length']
    train_test_params['reverse_seq_length_validation'] = config['reverse_seq_length_validation'] if 'reverse_seq_length_validation' in config else train_test_params['reverse_seq_length']
    if config['model_cls'] == 'MLP':
        model_params = dict(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim'],
            residuals=config['residuals'],
            dropout=config['dropout'] if 'dropout' in config else 0,
            activation=config['activation'],
            use_pre_layer_norm=config['use_pre_layer_norm'] if 'use_pre_layer_norm' in config else False,
            use_mean_and_scale=config['use_mean_and_scale'] if 'use_mean_and_scale' in config else False,
        )

    if config['model_cls'] == 'shPLRNN':
        model_params = dict(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
        )

    if config['model_cls'] == 'Transformer':
        model_params = dict(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            positional_embed=config['positional_embed'],
            max_len=config['max_len'],
            dropout=config['dropout'],
            activation=config['activation']
        )

    if config['model_cls'] == 'LSTM':
        model_params = dict(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim'],
            residuals=config['residuals'],
            activation=config['activation']
        )
    
    if config['model_cls'] == 'JacNet':
        model_params = dict(
            input_dim=config['input_dim'],
            embed_dim=config['embed_dim'],
            bottleneck_dim=config['bottleneck_dim'],
            jac_guess_rank=config['jac_guess_rank'],
            embedder=config['embedder'],
            use_resnet=config['use_resnet'],
            num_resnet_channels=config['num_resnet_channels'],
            num_resnet_layers=config['num_resnet_layers'],
            kernel_size=config['kernel_size'],
            activation=config['activation'],
            use_layer_norm=config['use_layer_norm']
        )
        model_params['embedder_kwargs'] = {key[9:]: val for key, val in config.items() if key.startswith('embedder_')}

    if config['model_cls'] == 'MatrixGenerator':
        model_params = dict(
            input_dim=config['input_dim'],
            model_type=config['model_type'],
            embedding_type=config['embedding_type'] if 'embedding_type' in config else 'integer',
        )
        model_params['model_kwargs'] = {key[6:]: val for key, val in config.items() if key.startswith('model_') and key != 'model_cls' and key != 'model_type' and key != 'model_obs_noise_scale'}

    model_params['_target_'] = 'CommunicationJacobians.models.' + config['model_cls'].lower() + '.' + config['model_cls']
    
    lightning_params = dict(
        _target_='CommunicationJacobians.models.' + config['model_cls'].lower() + '.' + config['lightning_cls'],
        direct=config['direct'],
        mode=config['mode'],
        int_method=config['int_method'],
        path=config['path'],
        discretization=config['discretization'] if 'discretization' in config else 'matrix_exp',
        loss_func=config['loss_func'],
        alpha_hal=config['alpha_hal'] if 'alpha_hal' in config else 0,
        loss_func_validation=config['loss_func_validation'] if 'loss_func_validation' in config else config['loss_func'],
        context_length=config['context_length'],
        interp_pts=config['interp_pts'],
        jac_penalty=config['jac_penalty'] if 'jac_penalty' in config else 0,
        jac_penalty_ord=config['jac_penalty_ord'] if 'jac_penalty_ord' in config else 'nuc',
        l2_penalty=config['l2_penalty'] if 'l2_penalty' in config else 0,
        path_point_mode=config['path_point_mode'] if 'path_point_mode' in config else 'interp',
        obs_noise_scale=config['obs_noise_scale'],
        target_smoothing_noise_scale=config['target_smoothing_noise_scale'] if 'target_smoothing_noise_scale' in config else 0,
        y0_noise_scale=config['y0_noise_scale'] if 'y0_noise_scale' in config else 0,
        noise_annealing=config['noise_annealing'] if 'noise_annealing' in config else False,
        # one_step_loss_weight=config['one_step_loss_weight'],
        # multi_step_loss_weight=config['multi_step_loss_weight'],
        # one_step_gen_loss_weight=config['one_step_gen_loss_weight'],
        # multi_step_gen_loss_weight=config['multi_step_gen_loss_weight'],
        # multi_step_spiral_loss_weight=config['multi_step_spiral_loss_weight'] if 'multi_step_spiral_loss_weight' in config else 0,
        # radius=config['radius'] if 'radius' in config else None,
        pre_epochs=config['pre_epochs'],
        model_obs_noise_scale=config['model_obs_noise_scale'],
        log_interval=config['log_interval'] if 'log_interval' in config else 1,
        jac_loss_interval=config['jac_loss_interval'] if 'jac_loss_interval' in config else 1,
        # line_vec=config['line_vec'] if 'line_vec' in config else False,
        # final_step_only=config['final_step_only'] if 'final_step_only' in config else False,
        # validate_all=config['validate_all'] if 'validate_all' in config else False,
        alpha_teacher_forcing=config['alpha_teacher_forcing'] if 'alpha_teacher_forcing' in config else 0,
        alpha_teacher_forcing_reverse=config['alpha_teacher_forcing_reverse'] if 'alpha_teacher_forcing_reverse' in config else 0,
        teacher_forcing_annealing=config['teacher_forcing_annealing'] if 'teacher_forcing_annealing' in config else False,
        gamma_teacher_forcing=config['gamma_teacher_forcing'] if 'gamma_teacher_forcing' in config else 0.999,
        gamma_teacher_forcing_reverse=config['gamma_teacher_forcing_reverse'] if 'gamma_teacher_forcing_reverse' in config else 0.999,
        teacher_forcing_update_interval=config['teacher_forcing_update_interval'] if 'teacher_forcing_update_interval' in config else 5,
        teacher_forcing_steps=config['teacher_forcing_steps'] if 'teacher_forcing_steps' in config else 1,
        min_alpha_teacher_forcing=config['min_alpha_teacher_forcing'] if 'min_alpha_teacher_forcing' in config else 0,
        alpha_validation=config['alpha_validation'] if 'alpha_validation' in config else 0,
        alpha_validation_reverse=config['alpha_validation_reverse'] if 'alpha_validation_reverse' in config else 0,
        random_points_interp_pts_validation=config['random_points_interp_pts_validation'] if 'random_points_interp_pts_validation' in config else 2,
        obs_noise_scale_validation=config['obs_noise_scale_validation'] if 'obs_noise_scale_validation' in config else 0,
        pre_train_epochs=config['pre_train_epochs'] if 'pre_train_epochs' in config else 0,
        reverse_weight=config['reverse_weight'] if 'reverse_weight' in config else 1,
        random_points_weight=config['random_points_weight'] if 'random_points_weight' in config else 1,
    )
    if 'learning_rate' in config:
        lightning_params['learning_rate'] = config['learning_rate']
    lightning_params['random_points_interp_pts'] = config['random_points_interp_pts'] if 'random_points_interp_pts' in config else config['interp_pts']
    lightning_params['max_random_points_interp_pts'] = config['max_random_points_interp_pts'] if 'max_random_points_interp_pts' in config else None
    lightning_params['random_points_interp_pts_validation'] = config['random_points_interp_pts_validation'] if 'random_points_interp_pts_validation' in config else config['random_points_interp_pts']

    logger_params = dict(
        _target_='pytorch_lightning.loggers.' + config['logger_cls'],
        save_dir=config['save_dir'],
        log_model=config['log_model']
    )

    trainer_params = dict(
        max_epochs=config['max_epochs'],
        limit_train_batches=config['limit_train_batches'],
        limit_val_batches=config['limit_val_batches'] if 'limit_val_batches' in config else 1.0,
        accumulate_grad_batches=config['accumulate_grad_batches'] if 'accumulate_grad_batches' in config else 1
    )

    ret_dict = {
        'data': {
            'data_type': data_type,
            'flow': flow_params,
            'trajectory_params': trajectory_params,
            'postprocessing': postprocessing,
            'train_test_params': train_test_params,
            'random_points_params': random_points_params,
            'random_points_reverse_params': random_points_reverse_params
        },
        'model': {
            'params': model_params
        },
        'training': {
            'batch_size': config['batch_size'],
            'save_top_k': config['save_top_k'] if 'save_top_k' in config else 3,
            'use_trajectory': config['use_trajectory'] if 'use_trajectory' in config else True,
            'use_random_points': config['use_random_points'] if 'use_random_points' in config else False,
            'reverse_training': config['reverse_training'] if 'reverse_training' in config else ('inverse_training' if 'inverse_training' in config else False),
            'reverse_validation': config['reverse_validation'] if 'reverse_validation' in config else ('inverse_validation' if 'inverse_validation' in config else False),
            'lightning': lightning_params,
            'logger': logger_params,
            'trainer_params': trainer_params
        }
    }
    
    return OmegaConf.create(ret_dict)