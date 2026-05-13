# Security Policy

## Reporting a Vulnerability

If you discover a security issue in ForestOptiLM / Nocturne Data Forge, please
report it **privately** through
[GitHub Security Advisories](https://github.com/therudywolf/ForestOptiLM/security/advisories/new).

Do **not** open a public issue for security vulnerabilities.

## Scope

This project runs locally and talks to a local LM Studio server. Security
concerns include:
- Accidental exposure of API keys or secrets in logs.
- Path-traversal when extracting archives.
- Unsafe deserialization of untrusted files.

We take these issues seriously and will respond as quickly as possible.
