#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fuzzy_tm_infer._logging import add_logging_args, configure_from_args, logger
else:
    from .._logging import add_logging_args, configure_from_args, logger


REPO_ROOT = (
    Path.cwd()
    if (Path.cwd() / "ansible" / "playbooks" / "rpi_fuzzy_tm_infer.yml").exists()
    else Path(__file__).resolve().parents[3]
)
DEFAULT_ENV_FILE = REPO_ROOT / "ansible" / "rpi.env"
PLAYBOOK = REPO_ROOT / "ansible" / "playbooks" / "rpi_fuzzy_tm_infer.yml"
REQUIREMENTS = REPO_ROOT / "ansible" / "requirements.yml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy and test fuzzy_tm_infer on Raspberry Pi via Ansible."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="path to env file with TM_RPI_HOST, TM_RPI_USER, and TM_RPI_PASSWORD",
    )
    parser.add_argument("--target", help="Ansible target alias, default from env file")
    parser.add_argument("--host", help="Raspberry Pi hostname/IP, default from env file")
    parser.add_argument("--user", help="SSH user, default from env file")
    parser.add_argument("--password", help="SSH/become password, default from env file")
    parser.add_argument("--archflags", help="native C ARCHFLAGS, default from env file")
    parser.add_argument("--no-assets", action="store_true", help="skip syncing assets/")
    parser.add_argument("--no-native-benchmark", action="store_true")
    parser.add_argument("--python-benchmark", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--booleanizer-benchmark", action="store_true")
    parser.add_argument(
        "--install-collections",
        action="store_true",
        help="run ansible-galaxy collection install before the playbook",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the ansible-playbook command without running it",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_from_args(args, default_level="INFO")

    env = _load_env(args.env_file)
    target = args.target or env.get("TM_RPI_TARGET", "rpi5")
    host = args.host or _env_value(env, "TM_RPI_HOST")
    user = args.user or _env_value(env, "TM_RPI_USER") or "rpi"
    password = args.password or _env_value(env, "TM_RPI_PASSWORD") or ""
    archflags = args.archflags or _env_value(env, "TM_RPI_ARCHFLAGS")
    remote_root = f"/home/{user}/{REPO_ROOT.name}"
    remote_venv = f"{remote_root}/.venv"

    if not host:
        raise SystemExit(
            f"Missing TM_RPI_HOST. Copy {DEFAULT_ENV_FILE}.example to {DEFAULT_ENV_FILE} "
            "or pass --host."
        )

    if args.install_collections:
        logger.info("installing Ansible collections from {}", REQUIREMENTS)
        subprocess.run(
            ["ansible-galaxy", "collection", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )

    with tempfile.NamedTemporaryFile("w", prefix="tm-rpi-", suffix=".ini", delete=False) as f:
        inventory = Path(f.name)
        f.write(_inventory_text(target, host, user, password))
    inventory.chmod(0o600)

    cmd = [
        "ansible-playbook",
        "-i",
        str(inventory),
        str(PLAYBOOK),
        "-e",
        f"tm_rpi_host={target}",
        "-e",
        f"tm_repo_local={REPO_ROOT}",
        "-e",
        f"tm_remote_root={remote_root}",
        "-e",
        f"tm_remote_venv={remote_venv}",
        "-e",
        f"sync_assets={str(not args.no_assets).lower()}",
        "-e",
        f"run_native_benchmark={str(not args.no_native_benchmark).lower()}",
        "-e",
        f"run_python_benchmark={str(args.python_benchmark).lower()}",
        "-e",
        f"run_compare_all={str(args.compare_all).lower()}",
        "-e",
        f"run_booleanizer_benchmark={str(args.booleanizer_benchmark).lower()}",
    ]
    if archflags:
        cmd.extend(["-e", f"tm_archflags={archflags}"])

    try:
        if args.dry_run:
            print(" ".join(shlex.quote(part) for part in cmd))
            return
        logger.info("running Ansible playbook for target {}", target)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    finally:
        with suppress(FileNotFoundError):
            inventory.unlink()


def _load_env(path: Path) -> dict[str, str]:
    values = dict(os.environ)
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _env_value(env: dict[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _inventory_text(target: str, host: str, user: str, password: str) -> str:
    lines = [
        "[rpi]",
        f"{target} ansible_host={host} ansible_user={user}",
        "",
        "[rpi:vars]",
        "ansible_python_interpreter=/usr/bin/python3",
    ]
    if password:
        lines.extend(
            [
                f"ansible_password={password}",
                f"ansible_become_password={password}",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
