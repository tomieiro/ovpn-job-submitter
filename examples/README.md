# Examples

## Run this first — no cluster needed

```bash
.venv/bin/python -m ipykernel install --user --name python3   # once
.venv/bin/python examples/local_dry_run.py
```

Runs the full `DGXClient` → `DGXJob.wait()` flow on your machine: builds a
real bundle from `experiment.ipynb`, then executes it with the exact same
`execute_notebook.py` runner that ships in the Docker image, using a plain
temp directory instead of SSH/SLURM/Docker. You'll see the same
`[Job <id>] STATE` and per-cell streaming output you'd get against a real
DGX. Good for sanity-checking a change to the library or seeing the API in
action without any cluster access.

## Against a real cluster

These need a real `.ovpn` file, a real cluster username, and
`DEFAULT_SSH_HOST` / `DEFAULT_REMOTE_BASE_DIR` in
[`src/dgx_slurm/client.py`](../src/dgx_slurm/client.py) set to your actual
DGX endpoint (they're placeholders in this repo). You'll be prompted for
your password via `getpass`.

- **`basic_usage.py`** — the canonical `submit()` + `await job.wait()` flow.
  ```bash
  python examples/basic_usage.py /path/to/client.ovpn cluster-user experiment.ipynb
  ```
- **`submit_with_includes.py`** — bundling extra data/source files alongside
  the notebook via `include=[...]`.
  ```bash
  python examples/submit_with_includes.py /path/to/client.ovpn cluster-user
  ```
- **`attach_to_job.py`** — reattaching to a job submitted in an earlier
  session, by SLURM job ID.
  ```bash
  python examples/attach_to_job.py /path/to/client.ovpn cluster-user 48192
  ```

## Files

- `experiment.ipynb` — the sample notebook submitted by the examples above.
  No GPU/heavy dependencies, so it also runs fine in `local_dry_run.py`.
