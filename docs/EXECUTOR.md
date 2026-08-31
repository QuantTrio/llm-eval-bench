# Remote executor

The executor runs untrusted benchmark code on a separate host through rootless Docker. It never
falls back to executing generated code on the machine running `llmbench`.

Build the sandbox and service images:

```bash
docker build -f executor/sandbox.Dockerfile -t ghcr.io/quanttrio/llmbench-sandbox:1.0.1 .
docker compose -f executor/docker-compose.yml build
```

The compose file expects a rootless Docker socket at `${XDG_RUNTIME_DIR}/docker.sock`. Put a TLS
reverse proxy in front of port 8765. `POST /v1/jobs` is rejected over plain HTTP unless
`allow_insecure: true` is explicitly selected for local development.

Networked tasks join the internal-only `llmbench-sandbox` network and can reach the internet only
through the Squid sidecar. Keep `executor.yaml` and `squid.conf` domain allowlists identical;
direct container egress remains unavailable.

A job request contains an ephemeral key and an allowlisted image:

```json
{
  "ephemeral_key": "short-lived-secret",
  "image": "ghcr.io/quanttrio/llmbench-sandbox:1.0.1",
  "command": ["-c", "print('ok')"],
  "network": false
}
```

The key is placed in a mode-0600 temporary env file, passed to the container as
`LLMBENCH_TASK_KEY`, redacted from output, and removed after completion. It is never part of the
stored job request, events, errors, or artifacts.
