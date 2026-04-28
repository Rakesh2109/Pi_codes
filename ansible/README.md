# Raspberry Pi Deployment

This folder deploys and tests `fuzzy_tm_infer` on Raspberry Pi 4/5.

Install Ansible requirements:

```bash
ansible-galaxy collection install -r ansible/requirements.yml
```

Create the local environment file:

```bash
cp ansible/rpi.env.example ansible/rpi.env
```

Edit `ansible/rpi.env`:

```text
TM_RPI_TARGET=rpi5
TM_RPI_HOST=<host-or-ip>
TM_RPI_USER=<ssh-user>
TM_RPI_PASSWORD=
TM_RPI_ARCHFLAGS=auto
```

`ansible/rpi.env` is ignored by git.

Preferred runner:

```bash
tm-rpi-deploy --install-collections
```

Run the full Python/native/Decision Tree comparison:

```bash
tm-rpi-deploy --compare-all
```

Run the booleanizer speed benchmark on the Pi:

```bash
tm-rpi-deploy --booleanizer-benchmark
```

Dry-run the generated command:

```bash
tm-rpi-deploy --dry-run
```

Useful variables:

```text
tm_archflags=auto
sync_assets=true
run_native_benchmark=true
run_python_benchmark=false
run_compare_all=false
run_booleanizer_benchmark=false
```

The playbook syncs the repo root, not only `src/fuzzy_tm_infer`, because
`pyproject.toml` and the source archives are repo-level files.

The remote root and virtual environment are derived automatically from the
remote SSH user's home directory and the local repository name.

When `TM_RPI_ARCHFLAGS=auto`, the playbook reads `/proc/device-tree/model` and
uses `-mcpu=cortex-a72` for Raspberry Pi 4, `-mcpu=cortex-a76` for Raspberry Pi
5, and `-mcpu=native` as a fallback.

Direct Ansible is still supported:

```bash
ansible-playbook -i ansible/inventory.rpi.example.ini \
  ansible/playbooks/rpi_fuzzy_tm_infer.yml
```
