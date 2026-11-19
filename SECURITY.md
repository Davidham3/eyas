# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities **privately**, not as a public issue: open
the repository's **Security** tab → "Report a vulnerability" (a private GitHub
security advisory).

We aim to acknowledge within 5 business days and will coordinate disclosure timing
with you. There is no bug bounty program.

## Scope

In scope: the framework code under `src/` and the shipped configs/scripts —
e.g. path traversal via run/config inputs, injection through the record-store or
trace files, credential handling in the OpenAI-compatible backend
(`api_key_env`).

Out of scope: vulnerabilities in the inference engine itself (vLLM — report
upstream), in the models you serve, or in datasets you bring. Note that
prompt optimization sends your task samples and seed prompts to the inference
endpoint you configure: treat that endpoint with the same trust level as the data,
and do not point Eyas at a third-party service with data you would not send it.
