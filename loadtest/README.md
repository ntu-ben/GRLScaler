# Load Testing Utilities

This directory contains scripts for running Locust-based load tests.

- `locust_agent.py` – REST API service to start Locust jobs remotely.
- `locust_agent_manual.py` – simple helper for manual long running tests.
- `onlineboutique/` – individual workload scenarios used by the above scripts.

## Manual Agent

`locust_agent_manual.py` launches Locust locally without the REST API. It
searches for the `locust` binary in `PATH` and starts the specified scenario in
headless mode. Logs and reports are written to `remote_logs/<tag>/`.

### Usage

```bash
python loadtest/locust_agent_manual.py --tag myrun \
       --scenario peak --target-host http://localhost:80 \
       --run-time 24h
```

- `--tag`        A folder name under `remote_logs/` for output.
- `--scenario`   One of the scripts in `onlineboutique/` without the prefix
                  `locust_` (e.g. `peak`, `offpeak`).
- `--target-host` Target service URL passed to Locust via `--host`.
- `--run-time`    Duration recognised by Locust; defaults to `24h`.

The script merely executes Locust and does not restart jobs. The chosen
scenario defines the traffic pattern. Each scenario now loops indefinitely so
that the run time is controlled solely by Locust's `--run-time` argument.
