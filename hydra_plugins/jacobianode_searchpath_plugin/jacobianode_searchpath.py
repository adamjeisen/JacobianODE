"""Hydra SearchPathPlugin that exposes JacobianODE-side conf YAMLs.

Hydra auto-discovers plugins by scanning the ``hydra_plugins`` namespace
package on import. Shipping this module under ``hydra_plugins/`` (rather
than under ``JacobianODE/``) is required for Hydra to pick it up — there
is no entry-point-based registration mechanism.

Once installed (via wheel or editable), consumer Hydra apps can do e.g.
``defaults: [override /model: latent_additive_coupling]`` against
JacobianODE-side config groups without manual ``hydra.searchpath`` setup.
"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class JacobianODESearchPathPlugin(SearchPathPlugin):
    """Make JacobianODE-side conf YAMLs discoverable in consumer projects."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="jacobianode",
            path="pkg://JacobianODE.jacobians.conf",
        )
