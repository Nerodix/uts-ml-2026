#!/usr/bin/env python3
"""Entrypoint perintah tunggal untuk mereproduksi seluruh pipeline UTS ML 2026.

Contoh penggunaan:
    python kode/run_all.py                 # Jalankan semua tahap + validasi output
    python kode/run_all.py --check-env     # Verifikasi environment, dataset, dan versi pustaka
    python kode/run_all.py --stages audit  # Jalankan tahap tertentu saja
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Pastikan direktori kode/ ada di sys.path saat dijalankan dari root repo maupun dari kode/
KODE_DIR = Path(__file__).resolve().parent
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from src.config import (  # noqa: E402
    CLUSTERING_PATH,
    DATA_DIR,
    KLASIFIKASI_TEST_PATH,
    KLASIFIKASI_TRAIN_PATH,
    REGRESI_TEST_PATH,
    REGRESI_TRAIN_PATH,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.utils import collect_environment_info, save_json, set_global_seed, timer  # noqa: E402
from src.validate_outputs import validate_all  # noqa: E402

AVAILABLE_STAGES = ["audit", "regresi", "klasifikasi", "clustering", "validate"]


def check_environment(save_report: bool = True) -> int:
    """Verifikasi ketersediaan dataset, versi pustaka, dan seed global."""
    ensure_dirs()
    env_info = collect_environment_info()
    required_files = [
        REGRESI_TRAIN_PATH,
        REGRESI_TEST_PATH,
        KLASIFIKASI_TRAIN_PATH,
        KLASIFIKASI_TEST_PATH,
        CLUSTERING_PATH,
    ]
    missing = [p.name for p in required_files if not p.exists()]
    env_info["missing_data_files"] = missing

    print(f"Python   : {env_info['python']}")
    print(f"Seed     : {SEED}")
    print(f"Data dir : {DATA_DIR.name}/")
    for pkg, ver in env_info["packages"].items():
        print(f"  - {pkg:<18} {ver}")

    if save_report:
        save_json(env_info, RESULTS_DIR / "environment.json")

    if missing:
        print("\nPERINGATAN: File dataset berikut belum ditemukan:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print("\nSemua file dataset dan pustaka utama siap.")
    return 0


def _run_python_script(rel_script: str, extra_args: list[str] | None = None) -> None:
    """Jalankan sub-script Python secara deterministik menggunakan interpreter aktif."""
    cmd = [sys.executable, str(KODE_DIR / rel_script)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def run_stage(stage: str) -> int:
    if stage == "regresi":
        _run_python_script("src/regresi/audit.py", ["--simpan-json"])
        _run_python_script("src/regresi/validation.py")
        _run_python_script("src/regresi/features.py", ["--uji-unseen"])
        _run_python_script("src/regresi/models.py")
        _run_python_script("src/regresi/calibration.py")
        _run_python_script("src/regresi/analysis.py")
        return 0

    if stage == "validate":
        return validate_all()

    raise NotImplementedError(
        f"Tahap '{stage}' belum diimplementasikan. Akan ditambahkan melalui PR masing-masing task."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-command runner UTS Machine Learning 2026.")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Cek ketersediaan dataset, versi pustaka, dan simpan results/environment.json.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=AVAILABLE_STAGES,
        default=AVAILABLE_STAGES,
        help="Tahap pipeline yang dijalankan.",
    )
    args = parser.parse_args()

    set_global_seed(SEED)
    ensure_dirs()

    if args.check_env:
        return check_environment(save_report=True)

    env_rc = check_environment(save_report=True)
    if env_rc != 0:
        return env_rc

    for stage in args.stages:
        with timer(f"Stage: {stage}"):
            rc = run_stage(stage)
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
