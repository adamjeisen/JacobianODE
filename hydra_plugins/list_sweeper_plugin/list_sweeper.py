# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified to support nested dict keys in list_params (flattened to dotted paths).

import copy
from dataclasses import dataclass

import itertools
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from hydra.types import HydraContext
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf, ListConfig

log = logging.getLogger(__name__)


def _flatten_dict(d, parent_key=""):
    """Recursively flatten a nested dict/DictConfig into dotted keys.

    Example:
        {'training': {'lightning': {'lr': [1, 2]}}}
        -> {'training.lightning.lr': [1, 2]}
    """
    items = {}
    mapping = d if isinstance(d, dict) else OmegaConf.to_container(d)
    for k, v in mapping.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


@dataclass
class LauncherConfig:
    _target_: str = (
        "hydra_plugins.list_sweeper_plugin.list_sweeper.ListSweeper"
    )
    list_params: DictConfig = DictConfig({})
    grid_params: DictConfig = DictConfig({})
    ablative_params: ListConfig = ListConfig([])


ConfigStore.instance().store(group="hydra/sweeper", name="list", node=LauncherConfig)


def flatten_tuple(original_tuple):
    new_tuple = ()
    for item in original_tuple:
        if isinstance(item, str):
            new_tuple += (item,)
        elif isinstance(item, list):
            new_tuple += tuple(item)
    return new_tuple


class ListSweeper(Sweeper):
    def __init__(self, list_params: DictConfig, grid_params: DictConfig, ablative_params: ListConfig):
        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None
        self.job_results = None
        self.list_params = list_params
        self.grid_params = grid_params
        self.ablative_params = ablative_params

    def setup(
            self,
            *,
            hydra_context: HydraContext,
            task_function: TaskFunction,
            config: DictConfig,
    ) -> None:
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context, task_function=task_function, config=config
        )
        self.hydra_context = hydra_context

    def sweep(self, arguments: List[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        print(f"Sweep output dir : {self.config.hydra.sweep.dir}")

        # Save sweep run config in top level sweep working directory
        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)
        grid_lists = []
        grid_keys = []
        # manage overrides
        for override in parsed:
            if override.is_sweep_override():
                sweep_choices = override.sweep_string_iterator()
                key = override.get_key_element()
                sweep = [f"{key}={val}" for val in sweep_choices]
                grid_lists.append(sweep)
            else:
                key = override.get_key_element()
                value = override.get_value_element_as_str()
                grid_lists.append([f"{key}={value}"])
            grid_keys.append(key)

        # manage grid params — flatten nested dicts to dotted keys
        for key, values in _flatten_dict(self.grid_params).items():
            values = self.parse(key, values)
            grid_lists.append([f"{key}={value}" for value in values])
            grid_keys.append(key)

        # manage list params — flatten nested dicts to dotted keys
        list_lists = []
        values_length = None
        for key, values in _flatten_dict(self.list_params).items():
            if key in grid_keys:
                log.warning(f"List key {key} is also a grid key. The list key will be ignored.")
                continue
            values = self.parse(key, values)
            # check if all lists have the same length
            if values_length is None:
                values_length = len(values)
            elif len(values) != values_length:
                raise ValueError(f"List key {key} has different length than other list keys")
            for idx, value in enumerate(values):
                if len(list_lists) <= idx:
                    list_lists.append([])
                list_lists[idx].append(f"{key}={value}")
        if len(list_lists) == 0:
            batch = list(itertools.product(*grid_lists))
        else:
            batch = list(itertools.product(*grid_lists, list_lists))
            # the list params are flattened to be part of the tuple
            batch = [flatten_tuple(x) for x in batch]

        # copy with ablative params
        if len(self.ablative_params) > 0:
            complete_batch = copy.deepcopy(batch)
            for ablative_dict in self.ablative_params:
                new_batch = copy.deepcopy(batch)
                new_batch = [list(x) for x in new_batch]
                for job in new_batch:
                    for key, value in ablative_dict.items():
                        found_key = False
                        for idx, key_param_str in enumerate(job):
                            job_key = key_param_str.split("=")[0]
                            if key == job_key:
                                found_key = True
                                job[idx] = f"{key}={ablative_dict[key]}"
                        if not found_key:
                            job.append(f"{key}={ablative_dict[key]}")
                    complete_batch.append(tuple(job))
            batch = complete_batch

        initial_job_idx = 0
        returns = [self.launcher.launch(batch, initial_job_idx)]
        return returns

    def parse(self, key, values):
        if isinstance(values, int) or isinstance(values, float) or isinstance(values, bool):
            values = [values]
        elif isinstance(values, str):
            if "," in values:
                values = values.replace(" ", "")
                values = values.replace("[", "")
                values = values.replace("]", "")
                values = values.split(",")
            else:
                values = [values]
        elif isinstance(values, (ListConfig, list)):
            values = list(values)
        else:
            raise ValueError(f"Cannot parse '{values}' for list key {key}")
        return values
