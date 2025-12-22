# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pkg_resources
from omegaconf import OmegaConf

from detectron2.config import LazyConfig


def try_get_key(cfg, *keys, default=None):
    """
    Try select keys from lazy cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def get_config(config_path):
    """
    Returns a config object from a config_path.

    Args:
        config_path (str): config file name relative to detrex's "configs/"
            directory, e.g., "common/train.py"

    Returns:
        omegaconf.DictConfig: a config object
    """
    # Try to get config file from pkg_resources (when detrex is installed as a package)
    cfg_file = None
    try:
    cfg_file = pkg_resources.resource_filename(
        "detrex.config", os.path.join("configs", config_path)
    )
        # Check if the file actually exists
        if not os.path.exists(cfg_file):
            cfg_file = None
    except (pkg_resources.DistributionNotFound, ModuleNotFoundError, KeyError):
        # Fallback to local file system (when detrex is used as local code)
        cfg_file = None
    except Exception:
        # Catch any other exceptions and fallback
        cfg_file = None
    
    # If pkg_resources didn't work, use local file system
    if cfg_file is None or not os.path.exists(cfg_file):
        # Get the directory where this config.py file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to detrex package root, then to configs directory
        # Structure: detrex/detrex/config/config.py -> detrex/configs/
        detrex_root = os.path.dirname(os.path.dirname(current_dir))
        configs_dir = os.path.join(detrex_root, "configs")
        cfg_file = os.path.join(configs_dir, config_path)
    
    if not os.path.exists(cfg_file):
        # Provide more detailed error message for debugging
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")
        raise RuntimeError(
            "{} not available in detrex configs!\n"
            "Tried path: {}\n"
            "Configs directory: {}\n"
            "Configs directory exists: {}".format(
                config_path, cfg_file, configs_dir, os.path.exists(configs_dir)
            )
        )
    cfg = LazyConfig.load(cfg_file)
    return cfg
