from datetime import datetime
import hydra
import numpy as np
import os
import pickle
import pytz
import torch
from sklearn.metrics import r2_score
import time
from tqdm.auto import tqdm
import wandb
from requests.exceptions import HTTPError
from .data_utils import compute_lyaps, estimate_weighted_jacobians, generate_train_and_test_sets, weighted_jacobian_lstsq
from .jacobian_utils import load_run, load_checkpoint


def collect_runs(model_name, cutoff_date, projects, obs_noises, network_info, loop_closure_weights, jac_penalties, seq_length=None, model_to_load=None, direct=None, run_number=0, use_base_deriv_pt=None, residuals=None, teacher_forcing_annealing=None, api=None, use_iterator=True):
    
    hyper_params_to_rerun = []

    runs_to_sweep = {}
    if use_iterator:
        iterator = tqdm(total=len(projects)*len(obs_noises)*len(network_info)*len(loop_closure_weights)*len(jac_penalties))
    else:
        iterator = None
    if api is None:
        api = wandb.Api()
    for project in projects:
        project_runs = api.runs(f"chaotic-consciousness/{project}")
        project_runs = [run for run in project_runs if datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC) > cutoff_date]
        for run in project_runs:
            if 'model' not in run.config:
                print(project)
                print(f"Run {run.id} has no model config")
                raise ValueError(f"Run {run.id} has no model config")
        project_runs = [run for run in project_runs if model_name in run.config['model']['params']['_target_']]

        if run_number == 0:
            project_runs = [run for run in project_runs if 'run_number' not in run.config['training'] or run.config['training']['run_number'] == run_number]
        else:
            project_runs = [run for run in project_runs if 'run_number' in run.config['training'] and run.config['training']['run_number'] == run_number]
        if model_to_load is not None:
            project_runs = [run for run in project_runs if run.config['data']['trajectory_params']['model_to_load'] == model_to_load]

        if seq_length is not None:
            project_runs = [run for run in project_runs if run.config['data']['train_test_params']['seq_length'] == seq_length]
        if direct is not None:
            project_runs = [run for run in project_runs if run.config['training']['lightning']['direct'] == direct]
        if use_base_deriv_pt is not None:
            project_runs = [run for run in project_runs if run.config['training']['lightning']['use_base_deriv_pt'] == use_base_deriv_pt]
        if residuals is not None:
            if model_name == 'MLP':
                project_runs = [run for run in project_runs if run.config['model']['params']['residuals'] == residuals]
            elif model_name == 'NeuralODE':
                project_runs = [run for run in project_runs if run.config['model']['params']['mlp_kwargs']['residuals'] == residuals]
        if teacher_forcing_annealing is not None:
            project_runs = [run for run in project_runs if run.config['training']['lightning']['teacher_forcing_annealing'] == teacher_forcing_annealing]

        runs_to_sweep[project] = {}
        for obs_noise in obs_noises:
            obs_noise_runs = [run for run in project_runs if np.abs(run.config['data']['postprocessing']['obs_noise'] - obs_noise) < 1e-3]
            runs_to_sweep[project][obs_noise] = []
            for net_size in network_info:
                hidden_dims = network_info[net_size]
                if model_name == 'MLP':
                    net_size_runs = [run for run in obs_noise_runs if run.config['model']['params']['hidden_dim'] == hidden_dims]
                elif model_name == 'NeuralODE':
                    net_size_runs = [run for run in obs_noise_runs if run.config['model']['params']['mlp_kwargs']['hidden_dim'] == hidden_dims]
                for loop_closure_weight in loop_closure_weights:
                    loop_closure_weight_runs = [run for run in net_size_runs if run.config['training']['lightning']['loop_closure_weight'] == loop_closure_weight]
                    for jac_penalty in jac_penalties:
                        jac_penalty_runs = [run for run in loop_closure_weight_runs if run.config['training']['lightning']['jac_penalty'] == jac_penalty]
                        if len(jac_penalty_runs) == 0:
                            print(f"No runs found for {project}, obs noise {obs_noise}, net size {net_size}, loop closure weight {loop_closure_weight}, jac penalty {jac_penalty}")
                            hyper_params_to_rerun.append((project, obs_noise, net_size, loop_closure_weight, jac_penalty))
                            if iterator is not None:
                                iterator.update(1)
                            continue
                        elif len(jac_penalty_runs) > 1:
                            print(f"Multiple runs found for {project}, obs noise {obs_noise}, net size {net_size}, loop closure weight {loop_closure_weight}, jac penalty {jac_penalty} ({[run.id for run in jac_penalty_runs]}), selecting first")
                            jac_penalty_run = jac_penalty_runs[0]
                        else:
                            jac_penalty_run = jac_penalty_runs[0]
                        
                        runs_to_sweep[project][obs_noise].append(dict(
                            id=jac_penalty_run.id,
                            project=project,
                            obs_noise=obs_noise,
                            net_size=net_size,
                            loop_closure_weight=loop_closure_weight,
                            jac_penalty=jac_penalty,
                        ))
                        if iterator is not None:
                            iterator.update(1)
    if iterator is not None:
        iterator.close()

    return runs_to_sweep, hyper_params_to_rerun

def sweep_runs(runs_to_sweep, projects, obs_noises, network_info, loop_closure_weights, jac_penalties, save_dir, use_loop_closure=True, use_jac_thresh=False, seq_length=None, eig_thresh=0.001, model_to_load=None, direct=None, use_base_deriv_pt=None, residuals=None, teacher_forcing_annealing=None, api=None):
    iterator = tqdm(total=len(projects)*len(obs_noises)*len(network_info))
    if api is None:
        api = wandb.Api()
    sweep_data = {}
    for project in projects:
        sweep_data[project] = {}
        project_runs = api.runs(f"chaotic-consciousness/{project}")
        for obs_noise in obs_noises:
            if seq_length is not None and seq_length != 25:
                seq_length_str = f"_seq_length_{seq_length}"
            else:
                seq_length_str = ""
            if model_to_load is not None:
                model_to_load_str = f"_model_to_load_{model_to_load}"
            else:
                model_to_load_str = ""
            if direct is not None:
                direct_str = f"_direct_{direct}"
            else:
                direct_str = ""
            if use_base_deriv_pt:
                use_base_deriv_pt_str = f"_use_base_deriv_pt_{use_base_deriv_pt}"
            else:
                use_base_deriv_pt_str = ""
            if residuals is not None:
                residuals_str = f"_residuals_{residuals}"
            else:
                residuals_str = ""
            if teacher_forcing_annealing is not None and not teacher_forcing_annealing:
                teacher_forcing_annealing_str = f"_teacher_forcing_annealing_{teacher_forcing_annealing}"
            else:
                teacher_forcing_annealing_str = ""
                
            save_path = os.path.join(save_dir, f"{project}_obs_noise_{obs_noise}_net_sizes_{'_'.join([str(size) for size in list(network_info.keys())])}_loop_closure_weights_{'_'.join([str(weight) for weight in list(loop_closure_weights)])}_jac_penalties_{'_'.join([str(penalty) for penalty in list(jac_penalties)])}{seq_length_str}{model_to_load_str}{direct_str}{use_base_deriv_pt_str}{residuals_str}{teacher_forcing_annealing_str}.pkl")
            if os.path.exists(save_path):
                sweep_data[project][obs_noise] = pickle.load(open(save_path, 'rb'))
                print(f"Loaded {project} {obs_noise} from {save_path}")
                iterator.update(1)
                continue
            else:
                print("--------------------------------")
                print(f"Project: {project}, Obs noise: {obs_noise}")
                print("--------------------------------")
                sweep_data[project][obs_noise] = {}
                # runs = [run for run in project_runs if run.id in [d['id'] for d in runs_to_sweep[project][obs_noise]]]
                runs = []
                for run in runs_to_sweep[project][obs_noise]:
                    run_id = run['id']
                    run_obj = api.run(f"chaotic-consciousness/{project}/{run_id}")
                    runs.append(run_obj)

                if len(runs) == 0:
                    print(f"No runs found for {project} with obs noise {obs_noise}")
                    continue
                print(f"Found {len(runs)} runs for {project} with obs noise {obs_noise}")

                run_ids = [run.id for run in runs]
                sweep_data[project][obs_noise]['run_ids'] = run_ids

                # ------------------------------------------------------------
                # get the run info
                # ------------------------------------------------------------
                run_info = {}
                for run in tqdm(runs):
                    epochs = [] 
                    val_jac_r2_scores = []
                    mean_val_losses = []
                    val_loop_closure_losses = []
                    for row in run.scan_history():
                        if 'val jac r2_score' in row and row['val jac r2_score'] is not None:
                            val_jac_r2_scores.append(row['val jac r2_score'])
                            epochs.append(row['epoch'])
                            mean_val_losses.append(row['mean val loss'])
                            if use_loop_closure:
                                val_loop_closure_losses.append(row['val loop closure loss'])
                        
                    
                    run_info[run.id] = {
                        'epochs': epochs,
                        'val_jac_r2_scores': val_jac_r2_scores,
                        'mean_val_losses': mean_val_losses,
                        'val_jac_r2_score': val_jac_r2_scores[np.argmin(mean_val_losses)],
                        'mean_val_loss': mean_val_losses[np.argmin(mean_val_losses)],
                        'val_loop_closure_loss': val_loop_closure_losses[np.argmin(mean_val_losses)] if use_loop_closure else None,
                        'epoch': epochs[np.argmin(mean_val_losses)]
                    }
                sweep_data[project][obs_noise]['run_info'] = run_info

                # ------------------------------------------------------------
                # get the loop info
                # ------------------------------------------------------------
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                n_batches = 100
                no_noise = False
                iterator_loop_info = tqdm(total=len(runs)*n_batches, desc='Computing batch info')
                loop_rets = {}
                print(runs)
                for run_ind, run in enumerate(runs):
                    if run_ind == 0:
                        generate_data = True
                        run, cfg, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model = load_run(project, run.id, no_noise=no_noise, save_dir=None, verbose=False, generate_data=generate_data)
                    else:
                        generate_data = False
                        run, cfg, _, _, _, _, _, _, _, lit_model = load_run(project, run_id=run.id, run=run, no_noise=no_noise, save_dir=None, verbose=False, generate_data=generate_data, dt=dt)

                    load_checkpoint(run, cfg, lit_model, save_dir=None, epoch=None, verbose=False)

                    lit_model.eval()
                    lit_model = lit_model.to(device)
                    jac_norms = []
                    one_step_pred_errors = []
                    batch_num = 0
                    num_eigs_too_low = 0
                    total_eigs = 0
                    with torch.no_grad():
                        for batch in val_dataloader:
                            if batch_num == n_batches:
                                break
                            batch = batch.to(device)
                            # loop_int, err_bound = loop_closure_with_est_error(batch, lit_model.compute_jacobians, dt=dt, n_loops=None, n_loop_pts=20, loop_path='line', int_method='Trapezoid', loop_closure_interp_pts=20, mix_trajectories=True, alpha=1, return_loop_pts=False, return_err_bound=True)
                            # loop_int_mses.append((loop_int**2).mean().cpu().float().item())
                            # err_bounds.append(err_bound.mean().cpu().float().item())
                            # error_ratios.append((torch.linalg.norm(loop_int, dim=-1)/err_bound).mean().float().item())
                            lit_model.dt = dt
                            batch_num += 1
                            pred_jacs = lit_model.compute_jacobians(batch)
                            eigs_pred = torch.linalg.eigvals(pred_jacs).real.flatten()
                            num_eigs_too_low += torch.sum(eigs_pred <= -1/dt).float().item()
                            total_eigs += len(eigs_pred)
                            jac_norm = torch.linalg.norm(pred_jacs, dim=(-2, -1)).mean().float().item()
                            jac_norms.append(jac_norm)
                            one_step_pred_errors.append(lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1)['loss'].float().item())
                            iterator_loop_info.update(1)
                    loop_rets[run.id] = {
                        'jac_norms': jac_norms,
                        'low_eigs_frac': num_eigs_too_low/total_eigs,
                        'one_step_pred_errors': one_step_pred_errors,
                        'mean_jac_norm': np.mean(jac_norms),
                        'mean_one_step_pred_error': np.mean(one_step_pred_errors),
                    }
                iterator_loop_info.close()
                
                # ------------------------------------------------------------
                # collect and filter runs
                # ------------------------------------------------------------
                mean_val_losses = np.zeros(len(runs))
                val_jac_r2_scores = np.zeros(len(runs))
                mean_jac_norms = np.zeros(len(runs))
                mean_one_step_pred_errors = np.zeros(len(runs))
                mean_val_loop_closure_losses = np.zeros(len(runs))
                low_eigs_fracs = np.zeros(len(runs))
                for i, run_id in enumerate(run_ids):
                    mean_val_losses[i] = run_info[run_id]['mean_val_loss']
                    val_jac_r2_scores[i] = run_info[run_id]['val_jac_r2_score']
                    mean_jac_norms[i] = loop_rets[run_id]['mean_jac_norm']
                    mean_one_step_pred_errors[i] = loop_rets[run_id]['mean_one_step_pred_error']
                    mean_val_loop_closure_losses[i] = run_info[run_id]['val_loop_closure_loss']
                    low_eigs_fracs[i] = loop_rets[run_id]['low_eigs_frac']
                sweep_data[project][obs_noise]['mean_val_losses'] = mean_val_losses
                sweep_data[project][obs_noise]['val_jac_r2_scores'] = val_jac_r2_scores
                sweep_data[project][obs_noise]['mean_jac_norms'] = mean_jac_norms
                sweep_data[project][obs_noise]['mean_one_step_pred_errors'] = mean_one_step_pred_errors
                sweep_data[project][obs_noise]['mean_val_loop_closure_losses'] = mean_val_loop_closure_losses
                sweep_data[project][obs_noise]['low_eigs_fracs'] = low_eigs_fracs

                persistence_baseline = ((trajs['train_trajs'].sequence[..., 1:, :] - trajs['train_trajs'].sequence[..., :-1, :])**2).mean()
                if use_jac_thresh:
                     # estimate local linear jacobians
                    jacs_pred_base, losses = estimate_weighted_jacobians(trajs['train_trajs'].sequence[:100], device='cuda', sweep=True, return_losses=True, discrete=False, dt=dt, verbose=True)
                    jac_thresh = torch.linalg.norm(jacs_pred_base, dim=(-2, -1)).mean().float().item()*1.1
                else:
                    jac_thresh = None

                dim = eq.embedding_dimension if cfg.data.data_type == 'dysts' else 128

                # usable_runs = mean_val_losses <= persistence_baseline.float().item()
                valid_models_bool0 = mean_one_step_pred_errors <= persistence_baseline.float().item()
                if np.sum(valid_models_bool0) == 0:
                    print(f"No runs with one step pred error less than persistence baseline.")
                    valid_models_bool0 = np.array([True]*len(runs))
                else:
                    print(f"Eliminated {np.sum(~valid_models_bool0)} runs with one step pred error greater than persistence baseline")
                print(f"mean_one_step_pred_errors: {mean_one_step_pred_errors}")
                print(f"persistence_baseline: {persistence_baseline.float().item()}")

                if use_loop_closure:
                    valid_models_bool1 = mean_val_loop_closure_losses <= np.sqrt(dim)
                    print(f"mean_val_loop_closure_losses: {mean_val_loop_closure_losses}")
                    print(f"np.sqrt(dim): {np.sqrt(dim)}")
                    if np.sum(valid_models_bool1 & valid_models_bool0) == 0:
                        print(f"All models w/ one step pred error less than persistence baseline are greater than {np.sqrt(dim)} loop closure loss")
                        valid_models_bool1 = np.array([True]*len(runs))
                    else:
                        # print(f"Eliminated {np.sum(~valid_models_bool1 & valid_models_bool0)} runs with loop closure loss greater than {np.sqrt(dim)}")
                        print(f"Eliminated {np.sum(~valid_models_bool1)} runs with loop closure loss greater than {np.sqrt(dim)}")
                else:
                    valid_models_bool1 = np.array([True]*len(runs))
                
                if use_jac_thresh:
                    valid_models_bool2 = mean_jac_norms <= jac_thresh
                    if np.sum(valid_models_bool2 & valid_models_bool0) == 0:
                        print(f"All models w/ one step pred error less than persistence baseline are greater than {jac_thresh} jacobian norm")
                        valid_models_bool2 = np.array([True]*len(runs))
                    else:
                        # print(f"Eliminated {np.sum(~valid_models_bool2 & valid_models_bool0)} runs with jacobian norm greater than {jac_thresh}")
                        print(f"Eliminated {np.sum(~valid_models_bool2)} runs with jacobian norm greater than {jac_thresh}")
                else:
                    valid_models_bool2 = np.array([True]*len(runs))
                
                valid_models_bool3 = low_eigs_fracs <= eig_thresh
                print(f"low eigs frac: {low_eigs_fracs}")
                print(f"val jac r2 score: {val_jac_r2_scores}")
                if np.sum(valid_models_bool3 & valid_models_bool0) == 0:
                    print(f"All models w/ one step pred error less than persistence baseline are greater than {eig_thresh} low eigs frac. Suggests noise is high, so ignore one step pred error filter.")
                    # valid_models_bool3 = np.array([True]*len(runs))
                    valid_models_bool0 = np.array([True]*len(runs))
                else:
                    # print(f"Eliminated {np.sum(~valid_models_bool3 & valid_models_bool0)} runs with low eigs frac greater than {eig_thresh}")
                    print(f"Eliminated {np.sum(~valid_models_bool3)} runs with low eigs frac greater than {eig_thresh}")
                
                valid_models_bool = valid_models_bool1 & valid_models_bool2 & valid_models_bool0 & valid_models_bool3
                model_inds = np.arange(len(runs))
                best_model = model_inds[valid_models_bool][np.argmin(mean_val_losses[valid_models_bool])]
            
                # find runs that were eliminated by error_ratio that were not eliminated by jac_norm
                # if use_loop_closure:
                #     eliminated_by_loop_closure_loss = np.where(~valid_models_bool1 & usable_runs)[0]
                #     eliminated_by_jac_norm = np.where(~valid_models_bool2 & usable_runs)[0]
                #     runs_eliminated_by_loop_closure_loss_but_not_jac_norm = np.setdiff1d(eliminated_by_loop_closure_loss, eliminated_by_jac_norm)
                #     print(f"Runs eliminated by loop closure loss but not jacobian norm: {runs_eliminated_by_loop_closure_loss_but_not_jac_norm}")
                #     sweep_data[project][obs_noise]['runs_eliminated_by_loop_closure_loss_but_not_jac_norm'] = runs_eliminated_by_loop_closure_loss_but_not_jac_norm

                sweep_data[project][obs_noise]['best_model'] = run_ids[best_model]
                sweep_data[project][obs_noise]['best_model_info'] = {
                    'mean_val_loss': mean_val_losses[best_model],
                    'val_jac_r2_score': val_jac_r2_scores[best_model],
                    'mean_jac_norm': mean_jac_norms[best_model],
                    'low_eigs_frac': low_eigs_fracs[best_model],
                    'net_size': runs_to_sweep[project][obs_noise][best_model]['net_size'],
                    'loop_closure_weight': runs_to_sweep[project][obs_noise][best_model]['loop_closure_weight'],
                    'jac_penalty': runs_to_sweep[project][obs_noise][best_model]['jac_penalty'],
                }
                print(sweep_data[project][obs_noise]['best_model_info'])

                sweep_data[project][obs_noise]['sweep_info'] = {
                    'persistence_baseline': persistence_baseline,
                    'jac_thresh': jac_thresh,
                    'valid_models_bool0': valid_models_bool0,
                    'valid_models_bool1': valid_models_bool1,
                    'valid_models_bool2': valid_models_bool2,
                    'valid_models_bool3': valid_models_bool3,
                }

                pickle.dump(sweep_data[project][obs_noise], open(save_path, 'wb'))
                torch.cuda.empty_cache()
                iterator.update(1)
    iterator.close()

    return sweep_data

def get_model_stuff(project, obs_noises, mlp_sweep_data, neuralode_sweep_data):
    model_ids = {}
    model_names = {}
    for obs_noise in obs_noises:
        model_ids[obs_noise] = []
        model_names[obs_noise] = {}
        mlp_run_id = mlp_sweep_data[project][obs_noise]['best_model']
        neuralode_run_id = neuralode_sweep_data[project][obs_noise]['best_model']
        model_ids[obs_noise].append(mlp_run_id)
        model_ids[obs_noise].append(neuralode_run_id)
        model_names[obs_noise][mlp_run_id] = f"JacobianODE ({obs_noise*100:.0f}% Noise)"
        model_names[obs_noise][neuralode_run_id] = f"NeuralODE ({obs_noise*100:.0f}% Noise)"
    
    model_data = {}
    all_trajs = {}
    for obs_noise in obs_noises:
        model_data[obs_noise] = {}
        for id_val in model_ids[obs_noise]:
            run, cfg, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model = load_run(project, id_val, no_noise=True, verbose=True)
            
            load_checkpoint(run, cfg, lit_model, save_dir=None, epoch=None, verbose=True)
            lit_model.eval()

            # Extract only necessary data from run object
            run_data = {
                'id': run.id,
                'name': run.name,
                'config': dict(run.config),  # Convert config to plain dict
            }

            model_data[obs_noise][id_val] = dict(
                run_data=run_data,  # Store extracted data instead of run object
                eq=eq,
                lit_model=lit_model,
            )
        all_trajs[obs_noise] = trajs
 
    return dict(model_data=model_data, model_names=model_names, model_ids=model_ids, cfg=cfg, eq=eq, values=values, all_trajs=all_trajs, dt=dt)

def get_model_traj_dict(model_data, model_names, model_ids, cfg, eq, values, all_trajs, dt, obs_noises):
    # ================================
    # Generate the test data
    # ================================
    if cfg.data.data_type == 'dysts':
        cfg.data.flow.random_state = 43
        eq_new = hydra.utils.instantiate(cfg.data.flow)
        cfg.data.trajectory_params.num_ics = 8
        cfg.data.trajectory_params.new_ic_mode = 'random'
        cfg.data.trajectory_params.traj_offset_sd = float(values.std())*0.1
        test_trajs = eq_new.make_trajectory(**cfg.data.trajectory_params)
        cfg.data.trajectory_params.num_ics = 32
        # obs_noise_scale = float(trajs['train_trajs'].sequence.std())*(0e-2)
        obs_noise_scale = 0
        cfg.data.flow.random_state = 42

        seq_length = 25
        train_dataset, val_dataset, test_dataset, _ = generate_train_and_test_sets(torch.from_numpy(test_trajs['values'] + np.random.randn(*test_trajs['values'].shape)*obs_noise_scale), seq_length, seq_spacing=10, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    else:
        test_trajs = all_trajs[obs_noises[0]]['test_trajs'].sequence
        obs_noise_scale = 0
        seq_length = 25
        _, _, test_dataset, _ = generate_train_and_test_sets(test_trajs + torch.randn_like(test_trajs)*obs_noise_scale, seq_length, seq_spacing=10, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    torch.manual_seed(42)
    np.random.seed(42)

    # ================================
    # Evaluate the models
    # ================================

    # for id_val in model_ids:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_batch_rets = {}
    for obs_noise in obs_noises:
        model_batch_rets[obs_noise] = {}
        for id_val in model_ids[obs_noise]:
            lit_model = model_data[obs_noise][id_val]["lit_model"]
            lit_model.eval()
            lit_model.to(device)
            with torch.no_grad():
                batch_rets = []
                for batch in tqdm(test_dataloader_new, desc=f"Evaluating {model_names[obs_noise][id_val]}"):
                # batch_n = 0
                # for batch in tqdm(val_dataloader):
                    # if batch_n > 100:
                    #     break
                    batch = batch.to(device)
                    if 'NeuralODE' in model_names[obs_noise][id_val]:
                        batch = batch[..., 15 - 1:, :]
                        ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.0, verbose=False, direct=False)
                    else:
                        ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.0, verbose=False, direct=True)
                    for key in ret.keys():
                        if key == 'metric_vals':
                            for key2 in ret[key].keys():
                                ret[key][key2] = ret[key][key2].cpu()
                        else:
                            ret[key] = ret[key].cpu().numpy()
                    # ret['mse'] = (batch - ret['outputs']).pow(2).mean()
                    batch_rets.append(ret)
                    # batch_n += 1
                model_batch_rets[obs_noise][id_val] = batch_rets
    
    # ================================
    # Compute the MSE and R2
    # ================================

    model_preds = {}
    for obs_noise in obs_noises:
        model_preds[obs_noise] = {}
        for model_id in model_ids[obs_noise]:
            mean_mse = 0
            mean_mase = 0
            mean_r2 = 0
            for batch_ret in model_batch_rets[obs_noise][model_id]:
                mean_mse += batch_ret['loss']
                mean_mase += batch_ret['metric_vals']['mase']
                mean_r2 += batch_ret['metric_vals']['r2_score']
            mean_mse /= len(model_batch_rets[obs_noise][model_id])
            mean_mase /= len(model_batch_rets[obs_noise][model_id])
            mean_r2 /= len(model_batch_rets[obs_noise][model_id])
            model_preds[obs_noise][model_id] = {
                'mse': mean_mse,
                'mase': mean_mase,
                'r2': mean_r2,
            }
            print(f"{model_names[obs_noise][model_id]}: MSE: {mean_mse}, MASE: {mean_mase}, R2: {mean_r2}")
    
    model_mses = {}
    model_r2s = {}
    for model_type in 'JacobianODE', 'NeuralODE':
        model_mses[model_type] = []
        model_r2s[model_type] = []
        for obs_noise in obs_noises:
            for model_id in model_ids[obs_noise]:
                if model_type in model_names[obs_noise][model_id]:
                    model_mses[model_type].append(model_preds[obs_noise][model_id]['mse'])
                    model_r2s[model_type].append(model_preds[obs_noise][model_id]['r2'])
                    break
        model_mses[model_type] = tuple(model_mses[model_type])
        model_r2s[model_type] = tuple(model_r2s[model_type])
    
    # model_mses = {
    #     obs_noise: tuple([model_preds[obs_noise][model_id]['mse'] for model_id in model_ids[obs_noise]]) for obs_noise in obs_noises
    # }

    return dict(model_preds=model_preds, model_mses=model_mses, model_r2s=model_r2s, test_trajs=test_trajs, test_dataloader_new=test_dataloader_new, model_batch_rets=model_batch_rets)

def get_jacs_pred_base(all_model_stuff, obs_noises, project):
    # ================================
    # Generate the test data
    # ================================
    cfg = all_model_stuff['cfg']
    values = all_model_stuff['values']
    all_trajs = all_model_stuff['all_trajs']
    if cfg.data.data_type == 'dysts':
        cfg.data.flow.random_state = 43
        eq_new = hydra.utils.instantiate(cfg.data.flow)
        cfg.data.trajectory_params.num_ics = 8
        cfg.data.trajectory_params.new_ic_mode = 'random'
        cfg.data.trajectory_params.traj_offset_sd = float(values.std())*0.1
        test_trajs = eq_new.make_trajectory(**cfg.data.trajectory_params)
        cfg.data.trajectory_params.num_ics = 32
        # obs_noise_scale = float(trajs['train_trajs'].sequence.std())*(0e-2)
        obs_noise_scale = 0
        cfg.data.flow.random_state = 42

        seq_length = 25
        train_dataset, val_dataset, test_dataset, _ = generate_train_and_test_sets(torch.from_numpy(test_trajs['values'] + np.random.randn(*test_trajs['values'].shape)*obs_noise_scale), seq_length, seq_spacing=10, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    else:
        test_trajs = all_trajs[obs_noises[0]]['test_trajs'].sequence
        obs_noise_scale = 0
        seq_length = 25
        _, _, test_dataset, _ = generate_train_and_test_sets(test_trajs + torch.randn_like(test_trajs)*obs_noise_scale, seq_length, seq_spacing=10, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)


    jacs_pred_base = {}

    for obs_noise in obs_noises:

        sample_model_id = list(all_model_stuff['mlp_model_data'][obs_noise].keys())[0]
        run, cfg, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model = load_run(project, sample_model_id, no_noise=False, verbose=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cfg.data.data_type == 'wmtask':
            _, losses, theta = estimate_weighted_jacobians(trajs['train_trajs'].sequence[:125], device=device, sweep=True, return_losses=True, return_theta=True, discrete=False, dt=dt, verbose=True)
        else:
            _, losses, theta = estimate_weighted_jacobians(trajs['train_trajs'].sequence, device=device, sweep=True, return_losses=True, return_theta=True, discrete=False, dt=dt, verbose=True)
        
        if cfg.data.data_type == 'dysts':
            train_plus_test_trajs = torch.cat((trajs['train_trajs'].sequence, torch.from_numpy(test_trajs['values']).float()))
        else:
            # train_plus_test_trajs = test_trajs
            train_plus_test_trajs = torch.cat((trajs['train_trajs'].sequence[:125], test_trajs[:125]))

        train_plus_test_trajs = train_plus_test_trajs.to(device)
        pairwise_dists = torch.cdist(train_plus_test_trajs, train_plus_test_trajs)
        d_vals = pairwise_dists.mean(axis=-1)
        lengthscales = d_vals/theta
        jacs_t_plus_t = weighted_jacobian_lstsq(train_plus_test_trajs, lengthscales, verbose=True)[0]
        if cfg.data.data_type == 'dysts':
            jacs_pred_base[obs_noise] = jacs_t_plus_t[-test_trajs['values'].shape[0]:].cpu()
        else:
            jacs_pred_base[obs_noise] = jacs_t_plus_t.cpu()[-125:]

    return jacs_pred_base

def get_model_jac_dict(model_data, model_names, model_ids, cfg, eq, values, all_trajs, dt, obs_noises, jacs_pred_base_all, norms_to_compute=['fro', 2], save_jacs=True):
    # ================================
    # Generate the test data
    # ================================

    if cfg.data.data_type == 'dysts':
        cfg.data.flow.random_state = 43
        eq_new = hydra.utils.instantiate(cfg.data.flow)
        cfg.data.trajectory_params.num_ics = 8
        cfg.data.trajectory_params.new_ic_mode = 'random'
        cfg.data.trajectory_params.traj_offset_sd = float(values.std())*0.1
        test_trajs = eq_new.make_trajectory(**cfg.data.trajectory_params)
        cfg.data.trajectory_params.num_ics = 32
        # obs_noise_scale = float(trajs['train_trajs'].sequence.std())*(0e-2)
        obs_noise_scale = 0
        cfg.data.flow.random_state = 42
        torch.manual_seed(42)
        np.random.seed(42)

        seq_length = 25
        train_dataset, val_dataset, test_dataset, _ = generate_train_and_test_sets(torch.from_numpy(test_trajs['values'] + np.random.randn(*test_trajs['values'].shape)*obs_noise_scale), seq_length, seq_spacing=seq_length, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    else:
        test_trajs = all_trajs[obs_noises[0]]['test_trajs'].sequence
        obs_noise_scale = 0
        seq_length = 25
        _, _, test_dataset, _ = generate_train_and_test_sets(test_trajs + torch.randn_like(test_trajs)*obs_noise_scale, seq_length, seq_spacing=seq_length, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # ================================
    # Evaluate the models
    # ================================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.data.data_type == 'dysts':
        jac_func_true = lambda x, t: torch.from_numpy(eq.jac(x.cpu().numpy(), t.cpu().numpy())).type(x.dtype).to(x.device)
    else:
        jac_func_true = lambda x, t: eq.jac(x)

    jac_data = {}

    if cfg.data.data_type == 'dysts':
        traj_batch = torch.from_numpy(test_trajs['values']).to(device)
    else:
        traj_batch = all_trajs[obs_noises[0]]['test_trajs'].sequence.to(device)

    # traj_batch = trajs['val_trajs'].sequence.to(device)
    jacs_true = jac_func_true(traj_batch, torch.arange(traj_batch.shape[-2])).cpu()

    # # jacs_pred_base, losses = estimate_weighted_jacobians(traj_batch.reshape(-1, traj_batch.shape[-1]), device=device, sweep=True, return_losses=True, discrete=False, dt=dt, verbose=True)
    # # jacs_pred_base = jacs_pred_base.reshape(traj_batch.shape[0], -1, traj_batch.shape[-1], traj_batch.shape[-1])
    # jacs_pred_base, losses = estimate_weighted_jacobians(traj_batch, device=device, sweep=True, return_losses=True, discrete=False, dt=dt, verbose=True)
    # jacs_pred_base = jacs_pred_base.cpu()

    for obs_noise in tqdm(obs_noises):
        jac_data[obs_noise] = {}

        for model_id in model_ids[obs_noise]:
            jac_data[obs_noise][model_id] = {}

            # jacs_pred_base, losses = estimate_weighted_jacobians(torch.from_numpy(test_trajs['values']).to(device), device=device, sweep=True, return_losses=True, discrete=False, dt=dt, verbose=True)

            # for batch in tqdm(test_dataloader_new):
            jac_data[obs_noise]['true'] = {}
            if save_jacs:
                jac_data[obs_noise]['true']['jacs'] = []
            jac_data[obs_noise]['true']['lyaps'] = []
            jac_data[obs_noise]['baseline'] = {}
            jac_data[obs_noise]['baseline']['mse'] = []
            for norm in norms_to_compute:
                jac_data[obs_noise]['baseline'][norm] = []
            jac_data[obs_noise]['baseline']['r2'] = []
            jac_data[obs_noise]['baseline']['lyaps'] = []
            if save_jacs:
                jac_data[obs_noise]['baseline']['jacs'] = []

            for model_id in model_ids[obs_noise]:
                jac_data[obs_noise][model_id] = {}
                jac_data[obs_noise][model_id]['mse'] = []
                for norm in norms_to_compute:
                    jac_data[obs_noise][model_id][norm] = []
                jac_data[obs_noise][model_id]['r2'] = []
                jac_data[obs_noise][model_id]['lyaps'] = []
                if save_jacs:
                    jac_data[obs_noise][model_id]['jacs'] = []

            for batch in [traj_batch]:
                if save_jacs:
                    jac_data[obs_noise]['true']['jacs'].append(jacs_true.cpu())
                batch = batch.to(device)
                # batch = batch + torch.randn_like(batch) * obs_noise * torch.linalg.norm(batch, dim=-1).mean() / np.sqrt(batch.shape[-1])
                for model_id in model_ids[obs_noise]:
                    
                    lit_model = model_data[obs_noise][model_id]["lit_model"]
                    lit_model.eval()
                    lit_model.to(device)
                    with torch.no_grad():
                        jacs_pred = torch.zeros(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[2])
                        for i in range(batch.shape[0]):
                            jacs_pred[i] = lit_model.compute_jacobians(batch[i].type(lit_model.dtype)).cpu()
                        mse = (jacs_pred - jacs_true).pow(2).mean()
                        for norm in norms_to_compute:
                            jac_data[obs_noise][model_id][norm].append(torch.linalg.norm(jacs_pred - jacs_true, ord=norm, dim=(-2, -1)).mean())
                        # r2 = 1 - (jacs_pred - jacs_true).pow(2).mean() / jacs_true.pow(2).mean()
                        r2 = r2_score(jacs_true.flatten(), jacs_pred.flatten())
                        jac_data[obs_noise][model_id]['mse'].append(mse)
                        jac_data[obs_noise][model_id]['r2'].append(r2)
                    jac_data[obs_noise][model_id]['lyaps'].append(compute_lyaps(torch.linalg.matrix_exp(jacs_pred*dt), dt=dt, verbose=False))
                    if save_jacs:   
                        jac_data[obs_noise][model_id]['jacs'].append(jacs_pred.cpu())
                
        # for batch in [traj_batch]:
        # jacs['baseline'].append(jacs_pred_base.cpu())
        # base_traj = torch.cat((all_trajs[obs_noise]['train_trajs'].sequence.to(device), traj_batch), dim=0)
        # jacs_pred_base, losses = estimate_weighted_jacobians(base_traj, device=device, sweep=True, return_losses=True, discrete=False, dt=dt, verbose=True)
        # jacs_pred_base = jacs_pred_base[all_trajs[obs_noise]['train_trajs'].sequence.shape[0]:].cpu()
        jacs_pred_base = jacs_pred_base_all[obs_noise]
        mse = (jacs_pred_base - jacs_true).pow(2).mean()
        # r2 = 1 - (jacs_pred_base.cpu() - jacs_true).pow(2).mean() / jacs_true.pow(2).mean()
        for norm in norms_to_compute:
            jac_data[obs_noise]['baseline'][norm].append(torch.linalg.norm(jacs_pred_base.cpu() - jacs_true, ord=norm, dim=(-2, -1)).mean())
        r2 = r2_score(jacs_true.flatten(), jacs_pred_base.flatten())
        if save_jacs:
            jac_data[obs_noise]['baseline']['jacs'].append(jacs_pred_base.cpu())
        jac_data[obs_noise]['baseline']['mse'].append(mse)
        jac_data[obs_noise]['baseline']['r2'].append(r2)
        jac_data[obs_noise]['baseline']['lyaps'].append(compute_lyaps(torch.linalg.matrix_exp(jacs_pred_base*dt), dt=dt, verbose=False))
        jac_data[obs_noise]['true']['lyaps'].append(compute_lyaps(torch.linalg.matrix_exp(jacs_true*dt), dt=dt, verbose=False))

    # ================================
    # Compute the MSE/Norm and R2
    # ================================

    jac_mses = {}
    jac_norms = {}
    for norm in norms_to_compute:
        jac_norms[norm] = {}
    jac_r2s = {}
    for model_type in 'JacobianODE', 'NeuralODE':
        jac_mses[model_type] = []
        jac_norms[model_type] = {}
        for norm in norms_to_compute:
            jac_norms[model_type][norm] = []
        jac_r2s[model_type] = []
        for obs_noise in obs_noises:
            for model_id in model_ids[obs_noise]:
                if model_type in model_names[obs_noise][model_id]:
                    jac_mses[model_type].append(np.mean(jac_data[obs_noise][model_id]['mse']))
                    jac_r2s[model_type].append(np.mean(jac_data[obs_noise][model_id]['r2']))
                    for norm in norms_to_compute:
                        jac_norms[model_type][norm].append(np.mean(jac_data[obs_noise][model_id][norm]))
                    break
        jac_mses[model_type] = tuple(jac_mses[model_type])
        jac_r2s[model_type] = tuple(jac_r2s[model_type])
        for norm in norms_to_compute:
            jac_norms[model_type][norm] = tuple(jac_norms[model_type][norm])
    
    # ================================
    # Compute Lyapunov MSE
    # ================================
    lyap_mses = {}
    for obs_noise in obs_noises:
        true_lyaps = np.vstack(jac_data[obs_noise]['true']['lyaps'])
        lyap_mses[obs_noise] = {}
        for model_id in model_ids[obs_noise]:
            model_lyaps = np.vstack(jac_data[obs_noise][model_id]['lyaps'])
            mse = ((true_lyaps - model_lyaps)**2).mean()
            lyap_mses[obs_noise][model_id] = mse
    
    return dict(jac_data=jac_data, jac_mses=jac_mses, jac_norms=jac_norms, jac_r2s=jac_r2s, lyap_mses=lyap_mses)

# CHECK FOR SIGTERM

def check_and_delete_log(run, verbose=False):
    # Download the log file
    log_file = run.file('output.log')

    log_file_down = log_file.download()
    
    # Read the contents directly from the file object
    content = log_file_down.read()
    
    # Check if "timed out" is in the content
    if "sigterm" in content.lower() or "timed out" in content.lower() or 'sigcont' in content.lower():
        if verbose:
            print("Found 'SIGTERM' in the log file. Deleting...")
        log_file_down.close()
        # delete the file
        os.remove("output.log")
        if verbose:
            print("File deleted successfully.")
        return True
    else:
        if verbose:
            print("No 'SIGTERM' found in the log file.")
        log_file_down.close()
        # delete the file
        os.remove("output.log")
        if verbose:
            print("File deleted anyway as per requirements.")
        return False

def check_for_sigterm(runs_to_sweep, projects, obs_noises, api=None, verbose=False):
    if api is None:
        api = wandb.Api()
    terminated_run_ids = []
    terminated_projects = []
    terminated_obs_noises = []
    terminated_jac_penalties = []
    terminated_loop_closure_weights = []

    total_its = 0
    for project in projects:
        for obs_noise in obs_noises:
            total_its += len(runs_to_sweep[project][obs_noise])

    iterator = tqdm(total=total_its)
    for project in projects:
        # project_runs = api.runs(f"chaotic-consciousness/{project}")
        for obs_noise in obs_noises:
            for chosen_run in runs_to_sweep[project][obs_noise]:
                # run = [run for run in project_runs if run.id == chosen_run['id']][0]
                run = api.run(f"chaotic-consciousness/{project}/{chosen_run['id']}")

                # 2 hours and 45 minutes
                runtime_seconds = 2*60*60 + 45*60
                
                if check_and_delete_log(run, verbose=verbose) and run.summary['_runtime'] <= runtime_seconds:
                    terminated_run_ids.append(chosen_run['id'])
                    terminated_projects.append(project)
                    terminated_obs_noises.append(obs_noise)
                    terminated_jac_penalties.append(chosen_run['jac_penalty'])
                    terminated_loop_closure_weights.append(chosen_run['loop_closure_weight'])
                iterator.update(1)
    iterator.close()

    return terminated_run_ids, terminated_projects, terminated_obs_noises, terminated_loop_closure_weights, terminated_jac_penalties

def get_all_runs(model_name, cutoff_date, projects, obs_noises, network_info, NUM_RUNS, mlp_sweep_data, seq_length=None, use_base_deriv_pt=None, residuals=None, api=None, model_to_load=None, teacher_forcing_annealing=None, direct=None):
    mlp_all_runs = {}
    mlp_hyper_params_to_rerun = []
    iterator = tqdm(total=len(projects)*len(obs_noises)*(NUM_RUNS))
    for project in projects:
        mlp_all_runs[project] = {}
        for obs_noise in obs_noises:
            mlp_all_runs[project][obs_noise] = []
            chosen_loop_closure_weight = mlp_sweep_data[project][obs_noise]['best_model_info']['loop_closure_weight']
            chosen_jac_penalty = mlp_sweep_data[project][obs_noise]['best_model_info']['jac_penalty']  # Replace 'pass' with the desired operation
            for run_number in range(NUM_RUNS):
                mlp_runs_to_sweep, hyper_params_to_rerun = collect_runs(model_name, cutoff_date, [project], [obs_noise], network_info, [chosen_loop_closure_weight], [chosen_jac_penalty], run_number=run_number, api=api, use_iterator=False, seq_length=seq_length, residuals=residuals, use_base_deriv_pt=use_base_deriv_pt, model_to_load=model_to_load, direct=direct, teacher_forcing_annealing=teacher_forcing_annealing)
                for i in range(len(hyper_params_to_rerun)):
                    hyper_params_to_rerun[i] = tuple(hyper_params_to_rerun[i]) + (run_number,)
                if len(hyper_params_to_rerun) > 0:
                    mlp_hyper_params_to_rerun.extend(hyper_params_to_rerun)
                else:
                    mlp_all_runs[project][obs_noise].extend(mlp_runs_to_sweep[project][obs_noise])
                iterator.update(1)
    iterator.close()

    return mlp_all_runs, mlp_hyper_params_to_rerun

def get_all_model_stuff(project, obs_noises, mlp_all_runs, neuralode_all_runs):
    mlp_model_data = {}
    neuralode_model_data = {}
    all_trajs = {}
    total_its = 0
    for obs_noise in obs_noises:
        total_its += len(mlp_all_runs[project][obs_noise])
        total_its += len(neuralode_all_runs[project][obs_noise])
    iterator = tqdm(total=total_its)
    for obs_noise in obs_noises:
        mlp_model_data[obs_noise] = {}
        neuralode_model_data[obs_noise] = {}
        for run_info in mlp_all_runs[project][obs_noise]:
            id_val = run_info['id']
            run, cfg, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model = load_run(project, id_val, no_noise=True, verbose=True)
            
            load_checkpoint(run, cfg, lit_model, save_dir=None, epoch=None, verbose=True)
            lit_model.eval()

            # Extract only necessary data from run object
            run_data = {
                'id': run.id,
                'name': run.name,
                'config': dict(run.config),  # Convert config to plain dict
            }

            mlp_model_data[obs_noise][id_val] = dict(
                run_data=run_data,  # Store extracted data instead of run object
                eq=eq,
                lit_model=lit_model,
            )
            iterator.update(1)
        all_trajs[obs_noise] = trajs

        for run_info in neuralode_all_runs[project][obs_noise]:
            id_val = run_info['id']
            run, cfg, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model = load_run(project, id_val, no_noise=True, verbose=True)
            
            load_checkpoint(run, cfg, lit_model, save_dir=None, epoch=None, verbose=True)
            lit_model.eval()

            neuralode_model_data[obs_noise][id_val] = dict(
                run_data=run_data,  # Store extracted data instead of run object
                eq=eq,
                lit_model=lit_model,
            )
            iterator.update(1)
    iterator.close()
    return dict(mlp_model_data=mlp_model_data, neuralode_model_data=neuralode_model_data, cfg=cfg, eq=eq, values=values, all_trajs=all_trajs, dt=dt)

def get_all_model_traj_dict(mlp_model_data, neuralode_model_data, cfg, eq, values, all_trajs, dt, obs_noises):
    # ================================
    # Generate the test data
    # ================================
    if cfg.data.data_type == 'dysts':
        cfg.data.flow.random_state = 43
        eq_new = hydra.utils.instantiate(cfg.data.flow)
        cfg.data.trajectory_params.num_ics = 8
        cfg.data.trajectory_params.new_ic_mode = 'random'
        cfg.data.trajectory_params.traj_offset_sd = float(values.std())*0.1
        test_trajs = eq_new.make_trajectory(**cfg.data.trajectory_params)
        cfg.data.trajectory_params.num_ics = 32
        # obs_noise_scale = float(trajs['train_trajs'].sequence.std())*(0e-2)
        obs_noise_scale = 0
        cfg.data.flow.random_state = 42

        seq_length = 25
        train_dataset, val_dataset, test_dataset, _ = generate_train_and_test_sets(torch.from_numpy(test_trajs['values'] + np.random.randn(*test_trajs['values'].shape)*obs_noise_scale), seq_length, seq_spacing=10, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    else:
        test_trajs = all_trajs[obs_noises[0]]['test_trajs'].sequence
        obs_noise_scale = 0
        seq_length = 25
        _, _, test_dataset, _ = generate_train_and_test_sets(test_trajs + torch.randn_like(test_trajs)*obs_noise_scale, seq_length, seq_spacing=10, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    torch.manual_seed(42)
    np.random.seed(42)

    # ================================
    # Evaluate the models
    # ================================
    total_its = 0
    for obs_noise in obs_noises:
        total_its += len(mlp_model_data[obs_noise])*len(test_dataloader_new)
        total_its += len(neuralode_model_data[obs_noise])*len(test_dataloader_new)
    iterator = tqdm(total=total_its, desc="Evaluating models")

    # for id_val in model_ids:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp_model_batch_rets = {}
    neuralode_model_batch_rets = {}
    for obs_noise in obs_noises:
        mlp_model_batch_rets[obs_noise] = {}
        neuralode_model_batch_rets[obs_noise] = {}
        # --- MLP ---
        for id_val in mlp_model_data[obs_noise].keys():
            lit_model = mlp_model_data[obs_noise][id_val]["lit_model"]
            lit_model.eval()
            lit_model.to(device)
            with torch.no_grad():
                batch_rets = []
                for batch in test_dataloader_new:
                # batch_n = 0
                # for batch in tqdm(val_dataloader):
                    # if batch_n > 100:
                    #     break
                    batch = batch.to(device)
                    ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.0, verbose=False, direct=True)
                    # if 'NeuralODE' in model_names[obs_noise][id_val]:
                    #     batch = batch[..., 15 - 1:, :]
                    #     ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.0, verbose=False, direct=False)
                    # else:
                    #     ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.0, verbose=False, direct=True)
                    for key in ret.keys():
                        if key == 'metric_vals':
                            for key2 in ret[key].keys():
                                ret[key][key2] = ret[key][key2].cpu()
                        else:
                            ret[key] = ret[key].cpu().numpy()
                    # ret['mse'] = (batch - ret['outputs']).pow(2).mean()
                    batch_rets.append(ret)
                    # batch_n += 1
                    iterator.update(1)
                mlp_model_batch_rets[obs_noise][id_val] = batch_rets
            lit_model = lit_model.to('cpu')
                
        # --- NeuralODE ---
        for id_val in neuralode_model_data[obs_noise].keys():
            lit_model = neuralode_model_data[obs_noise][id_val]["lit_model"]
            lit_model.eval()
            lit_model.to(device)
            with torch.no_grad():
                batch_rets = []
                for batch in test_dataloader_new:
                    batch = batch.to(device)
                    batch = batch[..., 15 - 1:, :]
                    ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.0, verbose=False, direct=False)
                    for key in ret.keys():
                        if key == 'metric_vals':
                            for key2 in ret[key].keys():
                                ret[key][key2] = ret[key][key2].cpu()
                        else:
                            ret[key] = ret[key].cpu().numpy()
                    batch_rets.append(ret)
                    iterator.update(1)
                neuralode_model_batch_rets[obs_noise][id_val] = batch_rets
            lit_model = lit_model.to('cpu')
                
    iterator.close()
    # ================================
    # Compute the MSE and R2
    # ================================

    mlp_model_preds = {}
    neuralode_model_preds = {}
    for obs_noise in obs_noises:
        mlp_model_preds[obs_noise] = {}
        neuralode_model_preds[obs_noise] = {}
        # --- MLP ---
        for model_id in mlp_model_data[obs_noise].keys():
            mean_mse = 0
            mean_mase = 0
            mean_r2 = 0
            for batch_ret in mlp_model_batch_rets[obs_noise][model_id]:
                mean_mse += batch_ret['loss']
                mean_mase += batch_ret['metric_vals']['mase']
                mean_r2 += batch_ret['metric_vals']['r2_score']
            mean_mse /= len(mlp_model_batch_rets[obs_noise][model_id])
            mean_mase /= len(mlp_model_batch_rets[obs_noise][model_id])
            mean_r2 /= len(mlp_model_batch_rets[obs_noise][model_id])
            mlp_model_preds[obs_noise][model_id] = {
                'mse': mean_mse,
                'mase': mean_mase,
                'r2': mean_r2,
            }
        # --- NeuralODE ---
        for model_id in neuralode_model_data[obs_noise].keys():
            mean_mse = 0
            mean_mase = 0
            mean_r2 = 0
            for batch_ret in neuralode_model_batch_rets[obs_noise][model_id]:
                mean_mse += batch_ret['loss']
                mean_mase += batch_ret['metric_vals']['mase']
                mean_r2 += batch_ret['metric_vals']['r2_score']
            mean_mse /= len(neuralode_model_batch_rets[obs_noise][model_id])
            mean_mase /= len(neuralode_model_batch_rets[obs_noise][model_id])
            mean_r2 /= len(neuralode_model_batch_rets[obs_noise][model_id])
            neuralode_model_preds[obs_noise][model_id] = {
                'mse': mean_mse,
                'mase': mean_mase,
                'r2': mean_r2,
            }

    return dict(mlp_model_preds=mlp_model_preds, neuralode_model_preds=neuralode_model_preds, test_trajs=test_trajs, test_dataloader_new=test_dataloader_new, mlp_model_batch_rets=mlp_model_batch_rets, neuralode_model_batch_rets=neuralode_model_batch_rets)

def get_all_model_jac_dict(mlp_model_data, neuralode_model_data, cfg, eq, values, all_trajs, dt, obs_noises, jacs_pred_base_all, norms_to_compute=['fro', 2], save_jacs=False):
    # ================================
    # Generate the test data
    # ================================

    if cfg.data.data_type == 'dysts':
        cfg.data.flow.random_state = 43
        eq_new = hydra.utils.instantiate(cfg.data.flow)
        cfg.data.trajectory_params.num_ics = 8
        cfg.data.trajectory_params.new_ic_mode = 'random'
        cfg.data.trajectory_params.traj_offset_sd = float(values.std())*0.1
        test_trajs = eq_new.make_trajectory(**cfg.data.trajectory_params)
        cfg.data.trajectory_params.num_ics = 32
        # obs_noise_scale = float(trajs['train_trajs'].sequence.std())*(0e-2)
        obs_noise_scale = 0
        cfg.data.flow.random_state = 42
        torch.manual_seed(42)
        np.random.seed(42)

        seq_length = 25
        train_dataset, val_dataset, test_dataset, _ = generate_train_and_test_sets(torch.from_numpy(test_trajs['values'] + np.random.randn(*test_trajs['values'].shape)*obs_noise_scale), seq_length, seq_spacing=seq_length, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    else:
        test_trajs = all_trajs[obs_noises[0]]['test_trajs'].sequence
        obs_noise_scale = 0
        seq_length = 25
        _, _, test_dataset, _ = generate_train_and_test_sets(test_trajs + torch.randn_like(test_trajs)*obs_noise_scale, seq_length, seq_spacing=seq_length, train_percent=0.0, test_percent=1.0, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False)
        test_dataloader_new = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # ================================
    # Evaluate the models
    # ================================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.data.data_type == 'dysts':
        jac_func_true = lambda x, t: torch.from_numpy(eq.jac(x.cpu().numpy(), t.cpu().numpy())).type(x.dtype).to(x.device)
    else:
        jac_func_true = lambda x, t: eq.jac(x)

    jac_data = {}

    if cfg.data.data_type == 'dysts':
        traj_batch = torch.from_numpy(test_trajs['values']).to(device)
    else:
        traj_batch = all_trajs[obs_noises[0]]['test_trajs'].sequence.to(device)

    # traj_batch = trajs['val_trajs'].sequence.to(device)
    jacs_true = jac_func_true(traj_batch, torch.arange(traj_batch.shape[-2])).cpu()

    # # jacs_pred_base, losses = estimate_weighted_jacobians(traj_batch.reshape(-1, traj_batch.shape[-1]), device=device, sweep=True, return_losses=True, discrete=False, dt=dt, verbose=True)
    # # jacs_pred_base = jacs_pred_base.reshape(traj_batch.shape[0], -1, traj_batch.shape[-1], traj_batch.shape[-1])
    # jacs_pred_base, losses = estimate_weighted_jacobians(traj_batch, device=device, sweep=True, return_losses=True, discrete=False, dt=dt, verbose=True)
    # jacs_pred_base = jacs_pred_base.cpu()

    total_its = 0
    for obs_noise in obs_noises:
        total_its += len(mlp_model_data[obs_noise])*1
        total_its += len(neuralode_model_data[obs_noise])*1
    iterator = tqdm(total=total_its, desc="Computing Jacobians")

    for obs_noise in obs_noises:
        jac_data[obs_noise] = {}

        # --- True and baseline ---
        jac_data[obs_noise]['true'] = {}
        if save_jacs:
            jac_data[obs_noise]['true']['jacs'] = []
            jac_data[obs_noise]['true']['jacs'].append(jacs_true.cpu())
        jac_data[obs_noise]['true']['lyaps'] = []
        jac_data[obs_noise]['baseline'] = {}
        jac_data[obs_noise]['baseline']['mse'] = []
        for norm in norms_to_compute:
            jac_data[obs_noise]['baseline'][norm] = []
        jac_data[obs_noise]['baseline']['r2'] = []
        jac_data[obs_noise]['baseline']['lyaps'] = []
        jacs_pred_base = jacs_pred_base_all[obs_noise]
        if save_jacs:
            jac_data[obs_noise]['baseline']['jacs'] = []
            jac_data[obs_noise]['baseline']['jacs'].append(jacs_pred_base.cpu())
        mse = (jacs_pred_base.cpu() - jacs_true[:jacs_pred_base.shape[0]]).pow(2).mean()
        # r2 = 1 - (jacs_pred_base.cpu() - jacs_true).pow(2).mean() / jacs_true.pow(2).mean()
        for norm in norms_to_compute:
            jac_data[obs_noise]['baseline'][norm].append(torch.linalg.norm(jacs_pred_base.cpu() - jacs_true[:jacs_pred_base.shape[0]], ord=norm, dim=(-2, -1)).mean())
        r2 = r2_score(jacs_true[:jacs_pred_base.shape[0]].flatten(), jacs_pred_base.flatten())
        jac_data[obs_noise]['baseline']['mse'].append(mse)
        jac_data[obs_noise]['baseline']['r2'].append(r2)

        jac_data[obs_noise]['baseline']['lyaps'].append(compute_lyaps(torch.linalg.matrix_exp(jacs_pred_base*dt), dt=dt, verbose=False))
        jac_data[obs_noise]['true']['lyaps'].append(compute_lyaps(torch.linalg.matrix_exp(jacs_true*dt), dt=dt, verbose=False))

        # --- MLP ---
        jac_data[obs_noise]['mlp'] = {}
        for model_id in mlp_model_data[obs_noise].keys():
            jac_data[obs_noise]['mlp'][model_id] = {}
            jac_data[obs_noise]['mlp'][model_id]['mse'] = []
            for norm in norms_to_compute:
                jac_data[obs_noise]['mlp'][model_id][norm] = []
            jac_data[obs_noise]['mlp'][model_id]['r2'] = []
            jac_data[obs_noise]['mlp'][model_id]['lyaps'] = []
            if save_jacs:
                jac_data[obs_noise]['mlp'][model_id]['jacs'] = []

            lit_model = mlp_model_data[obs_noise][model_id]["lit_model"]
            lit_model.eval()
            lit_model.to(device)
            for batch in [traj_batch]:
                
                batch = batch.to(device)
                # batch = batch + torch.randn_like(batch) * obs_noise * torch.linalg.norm(batch, dim=-1).mean() / np.sqrt(batch.shape[-1])
                    
                with torch.no_grad():
                    jacs_pred = torch.zeros(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[2])
                    for i in range(batch.shape[0]):
                        jacs_pred[i] = lit_model.compute_jacobians(batch[i].type(lit_model.dtype)).cpu()
                    mse = (jacs_pred - jacs_true).pow(2).mean()
                    for norm in norms_to_compute:
                        jac_data[obs_noise]['mlp'][model_id][norm].append(torch.linalg.norm(jacs_pred - jacs_true, ord=norm, dim=(-2, -1)).mean())
                    # r2 = 1 - (jacs_pred - jacs_true).pow(2).mean() / jacs_true.pow(2).mean()
                    r2 = r2_score(jacs_true.flatten(), jacs_pred.flatten())
                    jac_data[obs_noise]['mlp'][model_id]['mse'].append(mse)
                    jac_data[obs_noise]['mlp'][model_id]['r2'].append(r2)
                jac_data[obs_noise]['mlp'][model_id]['lyaps'].append(compute_lyaps(torch.linalg.matrix_exp(jacs_pred*dt), dt=dt, verbose=False))
                if save_jacs:   
                    jac_data[obs_noise]['mlp'][model_id]['jacs'].append(jacs_pred.cpu())
            lit_model = lit_model.to('cpu')
            iterator.update(1)
        # --- NeuralODE ---
        jac_data[obs_noise]['neuralode'] = {}
        for model_id in neuralode_model_data[obs_noise].keys():
            jac_data[obs_noise]['neuralode'][model_id] = {}
            jac_data[obs_noise]['neuralode'][model_id]['mse'] = []
            for norm in norms_to_compute:
                jac_data[obs_noise]['neuralode'][model_id][norm] = []
            jac_data[obs_noise]['neuralode'][model_id]['r2'] = []
            jac_data[obs_noise]['neuralode'][model_id]['lyaps'] = []
            if save_jacs:
                jac_data[obs_noise]['neuralode'][model_id]['jacs'] = []


            lit_model = neuralode_model_data[obs_noise][model_id]["lit_model"]
            lit_model.eval()
            lit_model.to(device)
            for batch in [traj_batch]:
                
                batch = batch.to(device)
                # batch = batch + torch.randn_like(batch) * obs_noise * torch.linalg.norm(batch, dim=-1).mean() / np.sqrt(batch.shape[-1])
                    
                with torch.no_grad():
                    jacs_pred = torch.zeros(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[2])
                    for i in range(batch.shape[0]):
                        jacs_pred[i] = lit_model.compute_jacobians(batch[i].type(lit_model.dtype)).cpu()
                    mse = (jacs_pred - jacs_true).pow(2).mean()
                    for norm in norms_to_compute:
                        jac_data[obs_noise]['neuralode'][model_id][norm].append(torch.linalg.norm(jacs_pred - jacs_true, ord=norm, dim=(-2, -1)).mean())
                    # r2 = 1 - (jacs_pred - jacs_true).pow(2).mean() / jacs_true.pow(2).mean()
                    r2 = r2_score(jacs_true.flatten(), jacs_pred.flatten())
                    jac_data[obs_noise]['neuralode'][model_id]['mse'].append(mse)
                    jac_data[obs_noise]['neuralode'][model_id]['r2'].append(r2)
                jac_data[obs_noise]['neuralode'][model_id]['lyaps'].append(compute_lyaps(torch.linalg.matrix_exp(jacs_pred*dt), dt=dt, verbose=False))
                if save_jacs:   
                    jac_data[obs_noise]['neuralode'][model_id]['jacs'].append(jacs_pred.cpu())
            lit_model = lit_model.to('cpu')
            iterator.update(1)
    iterator.close()
    return jac_data