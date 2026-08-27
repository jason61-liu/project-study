# E2B sandbox smoke test

This minimal script uses the official E2B Python SDK to create a sandbox from
`tpl-9f882069ddc2428bb5b62746`, execute a shell command through envd, and then
keep it available for 10 seconds for observation, and then destroy it.

Run it in the existing virtual environment:

```bash
source /root/workspace/pyproject/.venv/bin/activate
export E2B_API_KEY='<CubeAPI API key>'
python /root/workspace/pyproject/e2b-examples/agentScopeDemo/sandboxtest/e2b_sandbox_test.py
```

The connection defaults can be overridden with `E2B_API_URL`, `E2B_DOMAIN`,
`E2B_TEMPLATE_ID`, `E2B_LB_IP`, and `E2B_TIMEOUT_SECONDS`.

TLS verification uses the bundled `test-sandbox-ca.crt` by default. Set
`SSL_CERT_FILE` before starting the script to override it.
