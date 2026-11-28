# Security policy

## Supported versions

Security fixes go to the latest released minor version.

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through GitHub's
[security advisories](https://github.com/AdalbertoZambrini/camtensor/security/advisories/new)
("Report a vulnerability"). Include the version, the platform and, if
possible, a minimal reproduction.

You will get an acknowledgement within a few days. A fix and an advisory are
published once a patched release is available.

## Scope notes

- camtensor connects to RTSP servers and parses untrusted bitstreams; parser
  crashes or memory-safety problems in those paths are in scope.
- The service control plane (`camtensor serve`) has **no authentication** and
  binds to `127.0.0.1` by default. Do not expose it to untrusted networks
  without a reverse proxy that adds authentication.
- Credentials in stream URLs are redacted from statistics and logs; report
  any place where they leak.
