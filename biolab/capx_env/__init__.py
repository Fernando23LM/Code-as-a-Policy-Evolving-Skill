"""CaP-X bindings. Importing this package registers the environment, API and task.

CaP-X is not patched: its YAML configs name `biolab.capx_env.task.TubeRackCodeEnv`
as the env target, and importing that module runs the registrations below.
"""
from capx.envs.base import register_env
from capx.envs.tasks.base import CodeExecEnvConfig, register_config, register_exec_env
from capx.integrations.base_api import register_api

from .api import YamTubeRackPrivilegedApi
from .sim import TubeRackLowLevel
from .task import TubeRackCodeEnv

register_env("yam_tube_rack_low_level", TubeRackLowLevel)
register_api("YamTubeRackPrivilegedApi", YamTubeRackPrivilegedApi)
register_exec_env("yam_tube_rack_code_env", TubeRackCodeEnv)
register_config(
    "yam_tube_rack_code_env",
    CodeExecEnvConfig(low_level="yam_tube_rack_low_level", apis=["YamTubeRackPrivilegedApi"], privileged=True),
)

__all__ = ["TubeRackLowLevel", "YamTubeRackPrivilegedApi", "TubeRackCodeEnv"]
