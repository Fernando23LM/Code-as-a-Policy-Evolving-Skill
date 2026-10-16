"""CaP-X adapter for the plain API in biolab.api."""
from __future__ import annotations

from typing import Any

from capx.integrations.base_api import ApiBase

from ..api import TubeRackApi


class YamTubeRackPrivilegedApi(ApiBase):
    """Control of the two YAM arms plus ground-truth scene state."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self._api = TubeRackApi(env.sim)  # env.sim is reset in place, so this stays valid

    def functions(self) -> dict[str, Any]:
        return self._api.functions()
