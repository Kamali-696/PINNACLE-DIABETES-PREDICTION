"""Minimal DP-CTGAN-style training on fabricated data only.

The discriminator's real-record update uses Opacus DP-SGD with Poisson
subsampling, per-record clipping, Gaussian noise, and its RDP accountant.
The generator and fake-only discriminator update are post-processing of the
privatized discriminator and fresh randomness; neither reads real rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "hardened_run"
sys.path.insert(0, str(HERE / ".deps"))  # Isolated Opacus dependency only.

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from generate_fabricated_data import CATEGORIES, make_fabricated_data

SEED = 20261023
LATENT_DIM = 16
BATCH_SIZE = 64
EPOCHS = 8
NOISE_MULTIPLIER = 1.2
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
OUTPUT_ROWS = 500
DEVICE = torch.device("cpu")

BOUNDS = {"age": (18.0, 90.0), "bmi": (12.0, 60.0), "glucose": (50.0, 300.0)}
FEATURE_COLUMNS = ["age", "bmi", "glucose", "smoking_history", "diabetes"]


class Generator(nn.Module):
    """Mixed-type generator with fixed public transforms and no learned stats."""
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(LATENT_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.network(z)
        continuous = torch.tanh(raw[:, :3])
        smoking = torch.softmax(raw[:, 3:7], dim=1)
        diabetes = torch.sigmoid(raw[:, 7:8])
        return torch.cat([continuous, smoking, diabetes], dim=1)


class Discriminator(nn.Module):
    """Per-row MLP; no batch normalization or cross-record operations."""
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    def forward(self, row: torch.Tensor) -> torch.Tensor:
        return self.network(row).reshape(-1)


def encode_frame(frame: pd.DataFrame) -> np.ndarray:
    """Apply only fixed bounds and a predeclared category vocabulary."""
    encoded = np.empty((len(frame), 8), dtype=np.float32)
    for i, col in enumerate(("age", "bmi", "glucose")):
        low, high = BOUNDS[col]
        encoded[:, i] = 2.0 * (frame[col].to_numpy(dtype=np.float32) - low) / (high - low) - 1.0
    category_values = frame["smoking_history"].astype(str).to_numpy()
    category_index = {category: i for i, category in enumerate(CATEGORIES)}
    if not set(category_values).issubset(category_index):
        raise ValueError("smoking_history contains a value outside the fixed public vocabulary")
    encoded[:, 3:7] = 0.0
    encoded[np.arange(len(frame)), 3 + np.array([category_index[x] for x in category_values])] = 1.0
    encoded[:, 7] = frame["diabetes"].to_numpy(dtype=np.float32)
    return encoded


def decode_generated(values: np.ndarray) -> pd.DataFrame:
    """Convert model outputs to the original toy schema with fixed bounds."""
    values = np.asarray(values)
    result: dict[str, object] = {}
    for i, col in enumerate(("age", "bmi", "glucose")):
        low, high = BOUNDS[col]
        result[col] = low + (np.clip(values[:, i], -1, 1) + 1) * 0.5 * (high - low)
    result["smoking_history"] = [CATEGORIES[i] for i in np.argmax(values[:, 3:7], axis=1)]
    result["diabetes"] = (values[:, 7] >= 0.5).astype(np.int8)
    return pd.DataFrame(result, columns=FEATURE_COLUMNS)


def _accountant_steps(privacy_engine: PrivacyEngine) -> int:
    # Opacus RDP accountant history stores (noise_multiplier, sample_rate, steps).
    return int(sum(int(entry[2]) for entry in privacy_engine.accountant.history))


def main() -> None:
    if torch.__version__.split("+")[0] != "2.14.0":
        raise RuntimeError(f"Expected inspected PyTorch 2.14.0, found {torch.__version__}")
    import opacus
    if opacus.__version__ != "1.6.0":
        raise RuntimeError(f"Expected isolated Opacus 1.6.0, found {opacus.__version__}")

    # Fail closed before reading the fabricated input if Opacus cannot create
    # its configured cryptographic RNG. On this Windows runtime Opacus 1.6.0
    # requires torchcsprng and hardcodes /dev/urandom in its secure-mode setup.
    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=True)

    torch.use_deterministic_algorithms(True)

    data_path = HERE / "fabricated_data.csv"
    if not data_path.exists():
        raise FileNotFoundError("Run generate_fabricated_data.py first; this script reads no other data path.")
    frame = pd.read_csv(data_path)
    if len(frame) != 500 or frame.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Fabricated input must have exactly 500 rows and the expected five-column schema")
    if frame.isna().any().any():
        raise ValueError("Toy input contains missing values")
    x = encode_frame(frame)
    if not np.isfinite(x).all():
        raise ValueError("Encoded toy data contains non-finite values")

    # Do not seed sampling. In secure mode, Opacus replaces the sampler RNG
    # with its CSPRNG; predictable Poisson sampling would invalidate the
    # subsampling amplification assumed by the accountant.
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=False)

    # Fixed model initialization is isolated from the global torch RNG so it
    # does not make Opacus's subsequently drawn DP noise predictable.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        generator = Generator().to(DEVICE)
        discriminator = Discriminator().to(DEVICE)
    latent_rng = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    # Separate optimizers: real-record gradients get DP-SGD; fake-only updates
    # use generated samples and are post-processing of the private model state.
    private_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    fake_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)

    private_discriminator, private_optimizer, private_loader = privacy_engine.make_private(
        module=discriminator,
        optimizer=private_optimizer,
        data_loader=loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
        poisson_sampling=True,
    )
    actual_sample_rate = float(private_loader.sample_rate)
    sampler_rng = private_loader.batch_sampler.generator
    if sampler_rng is not privacy_engine.secure_rng:
        raise RuntimeError("Poisson sampler is not using Opacus secure_rng")
    if not private_optimizer.secure_mode or private_optimizer.generator is not privacy_engine.secure_rng:
        raise RuntimeError("DPOptimizer is not using Opacus secure_rng in secure mode")
    if actual_sample_rate <= 0 or actual_sample_rate > 1:
        raise RuntimeError(f"Invalid Opacus sampling rate: {actual_sample_rate}")

    private_update_calls = 0
    real_loss_sum = 0.0
    fake_loss_sum = 0.0
    generator_loss_sum = 0.0
    nonempty_batches = 0
    for _epoch in range(EPOCHS):
        for (real_batch,) in private_loader:
            real_batch = real_batch.to(DEVICE)
            if real_batch.shape[0] == 0:
                # Empty Poisson batches do not trigger an optimizer/accountant step.
                continue

            # One per-record loss per selected real record; Opacus clips these
            # per-sample gradients, adds Gaussian noise, and accounts on step().
            private_optimizer.zero_grad(set_to_none=True)
            real_scores = private_discriminator(real_batch)
            real_loss = F.softplus(-real_scores).mean()
            real_loss.backward()
            private_optimizer.step()
            private_update_calls += 1
            real_loss_sum += float(real_loss.detach().cpu())

            # These passes use generated rows and the already privatized model;
            # they are post-processing. Disable Opacus per-sample hooks so they
            # cannot be mistaken for additional real-data gradient contributions.
            private_discriminator.disable_hooks()
            try:
                with torch.no_grad():
                    fake_batch = generator(torch.randn(real_batch.shape[0], LATENT_DIM, device=DEVICE, generator=latent_rng))
                fake_optimizer.zero_grad(set_to_none=True)
                fake_scores = discriminator(fake_batch.detach())
                fake_loss = F.softplus(fake_scores).mean()
                fake_loss.backward()
                fake_optimizer.step()
                fake_loss_sum += float(fake_loss.detach().cpu())

                # Generator update is post-processing of the discriminator state.
                generator_optimizer.zero_grad(set_to_none=True)
                generated = generator(torch.randn(real_batch.shape[0], LATENT_DIM, device=DEVICE, generator=latent_rng))
                generator_loss = F.softplus(-discriminator(generated)).mean()
                generator_loss.backward()
                generator_optimizer.step()
                generator_loss_sum += float(generator_loss.detach().cpu())
                nonempty_batches += 1
            finally:
                private_discriminator.enable_hooks()

    accountant_steps = _accountant_steps(privacy_engine)
    if private_update_calls != accountant_steps:
        raise RuntimeError(f"Accountant counted {accountant_steps} updates but loop executed {private_update_calls}")
    history = privacy_engine.accountant.history
    if not history:
        raise RuntimeError("Opacus accountant recorded no private updates")
    if any(not np.isclose(float(entry[0]), NOISE_MULTIPLIER) or not np.isclose(float(entry[1]), actual_sample_rate)
           for entry in history):
        raise RuntimeError(f"Accountant events do not match configured noise/sample rate: {history}")
    epsilon = float(privacy_engine.get_epsilon(delta=DELTA))

    with torch.no_grad():
        generated = generator(torch.randn(OUTPUT_ROWS, LATENT_DIM, device=DEVICE, generator=latent_rng)).cpu().numpy()
    synthetic = decode_generated(generated)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_path = OUTPUT_DIR / "synthetic_fabricated.csv"
    synthetic.to_csv(synthetic_path, index=False)

    validation = {
        "rows": int(len(synthetic)),
        "schema_matches": synthetic.columns.tolist() == FEATURE_COLUMNS,
        "missing_cells": int(synthetic.isna().sum().sum()),
        "age_in_bounds": bool(synthetic.age.between(*BOUNDS["age"]).all()),
        "bmi_in_bounds": bool(synthetic.bmi.between(*BOUNDS["bmi"]).all()),
        "glucose_in_bounds": bool(synthetic.glucose.between(*BOUNDS["glucose"]).all()),
        "categories_valid": bool(synthetic.smoking_history.isin(CATEGORIES).all()),
        "binary_target_valid": bool(synthetic.diabetes.isin([0, 1]).all()),
    }
    if not all(v for k, v in validation.items() if isinstance(v, bool)):
        raise RuntimeError(f"Synthetic validation failed: {validation}")

    report = {
        "status": "TOY RUN COMPLETE; CRYPTOGRAPHIC RNG PATH ASSERTED",
        "data_origin": "500 fabricated records from generate_fabricated_data.py only",
        "training_rows": int(len(frame)),
        "synthetic_rows": int(len(synthetic)),
        "mechanism": "Opacus 1.6.0 DP-SGD on the discriminator real-record update; generator and fake-only discriminator updates are post-processing of DP state and fresh randomness.",
        "privacy_unit": "one fabricated row (record-level adjacency); no patient-level claim",
        "accountant": "Opacus RDP accountant, measured after the actual loop",
        "epsilon_measured": epsilon,
        "delta_configured": DELTA,
        "noise_multiplier": NOISE_MULTIPLIER,
        "max_grad_norm": MAX_GRAD_NORM,
        "sampling": "Poisson subsampling via Opacus DPDataLoader",
        "sample_rate_per_update": actual_sample_rate,
        "private_updates_executed": private_update_calls,
        "accountant_updates": accountant_steps,
        "accountant_history": [list(map(float, entry[:2])) + [int(entry[2])] for entry in history],
        "rng_mode": "Opacus secure_mode=True; sampler and DPOptimizer are configured to share the Opacus torchcsprng random-device generator",
        "preprocessing": "fixed public bounds and fixed category vocabulary; no data-derived scaling, imputers, category frequencies, or condition sampling",
        "postprocessing_validation": validation,
        "limitations": [
            "secure-mode guarantees depend on a correctly built torchcsprng and trusted OS random device",
            "fixed seeds apply to fabricated rows, initialization, and latent samples; DP noise is intentionally not fixed-seed reproducible",
            "record-level adjacency only; patient identity/unit not established",
            "toy utility and clinical validity are not assessed",
            "this is a minimal CTGAN-style GAN, not a faithful reproduction of the medical paper implementation",
        ],
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "opacus": opacus.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (OUTPUT_DIR / "privacy_accounting_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    privacy_text = (
        "DP-CTGAN-style toy privacy accounting report\n"
        "=============================================\n"
        f"Status: {report['status']}\n"
        "Data: fabricated toy rows only; no project datasets accessed\n"
        "Privacy unit: one toy row (record-level adjacency)\n"
        "Mechanism: Opacus 1.6.0 DP-SGD on real-record discriminator updates;\n"
        "generator and fake-only discriminator steps are post-processing\n"
        "Accountant: Opacus RDP, measured after training\n"
        f"Measured epsilon: {epsilon:.12f}\n"
        f"Configured delta: {DELTA}\n"
        f"Noise multiplier: {NOISE_MULTIPLIER}\n"
        f"Per-record clipping norm: {MAX_GRAD_NORM}\n"
        "Sampling: Poisson via Opacus DPDataLoader\n"
        f"Actual sampling rate per private update: {actual_sample_rate}\n"
        f"Private updates executed: {private_update_calls}\n"
        f"Accountant updates: {accountant_steps}\n"
        f"Accountant events (noise multiplier, sample rate, count): {history}\n"
        "Preprocessing: fixed bounds/category vocabulary only; no learned\n"
        "statistics, category frequencies, imputers, or conditional frequencies\n"
        "Secure randomness: Opacus secure_mode=True; sampler RNG identity checked\n"
        "Limitations: toy rows only;\n"
        "record-level only; not a medical or production privacy claim\n"
    )
    (OUTPUT_DIR / "privacy_accounting_report.txt").write_text(privacy_text, encoding="utf-8")
    (OUTPUT_DIR / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    training_report = {
        "status": "completed on fabricated toy data; DP noise intentionally not deterministic",
        "training_rows": int(len(frame)),
        "epochs": EPOCHS,
        "nonempty_poisson_batches": nonempty_batches,
        "private_discriminator_updates": private_update_calls,
        "mean_real_loss": real_loss_sum / max(private_update_calls, 1),
        "mean_fake_loss": fake_loss_sum / max(nonempty_batches, 1),
        "mean_generator_loss": generator_loss_sum / max(nonempty_batches, 1),
        "synthetic_rows": int(len(synthetic)),
        "all_training_data_fabricated": True,
    }
    (OUTPUT_DIR / "training_report.json").write_text(json.dumps(training_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
