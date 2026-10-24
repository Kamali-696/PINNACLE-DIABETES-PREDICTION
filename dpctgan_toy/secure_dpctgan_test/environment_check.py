"""Read-only, data-free runtime and accountant compatibility probe."""
from __future__ import annotations

import importlib.util
import json
import platform
import sys

import torch
import opacus
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant


def main() -> None:
    cfg = {"sample_rate": 0.125, "noise_multiplier": 1.2, "steps": 64, "delta": 1e-5}
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "opacus": opacus.__version__,
        "torchcsprng_installed": importlib.util.find_spec("torchcsprng") is not None,
        "accounting_configuration": cfg,
    }

    try:
        engine = PrivacyEngine(accountant="rdp", secure_mode=True)
        result["secure_mode_initialized"] = True
        result["secure_rng_type"] = type(engine.secure_rng).__name__
    except Exception as exc:  # Record the exact startup failure; do not fall back.
        result["secure_mode_initialized"] = False
        result["secure_mode_error"] = f"{type(exc).__name__}: {exc}"

    opacus_accountant = RDPAccountant()
    for _ in range(cfg["steps"]):
        opacus_accountant.step(
            noise_multiplier=cfg["noise_multiplier"],
            sample_rate=cfg["sample_rate"],
        )
    result["opacus_rdp_replay_epsilon"] = opacus_accountant.get_epsilon(cfg["delta"])

    try:
        from dp_accounting import dp_event, rdp

        independent = rdp.RdpAccountant()
        event = dp_event.PoissonSampledDpEvent(
            cfg["sample_rate"], dp_event.GaussianDpEvent(cfg["noise_multiplier"])
        )
        independent.compose(event, count=cfg["steps"])
        result["google_dp_accounting_rdp_epsilon"] = independent.get_epsilon(cfg["delta"])
        result["independent_accountant"] = "Google dp-accounting 0.6.0 RDP; PoissonSampledDpEvent of GaussianDpEvent, composed 64 times"
        result["accountant_numeric_difference"] = abs(
            result["opacus_rdp_replay_epsilon"] - result["google_dp_accounting_rdp_epsilon"]
        )
    except Exception as exc:
        result["independent_accountant_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
