# Domain recovery

The owner-maintained domain recovery bundle is stored in Dropbox:

```text
~/Dropbox/Work/infrastructure/domains/README.md
~/Dropbox/Work/infrastructure/domains/xuananh1.site/
```

Open the top-level README for the index, then `xuananh1.site/DOMAIN-USAGE.md` for the hostname-to-repository mapping, application settings and connector management links. `xuananh1.site/README.md` explains registrar transfers, new-domain migration and verification. The inventory was audited on 2026-10-28; refresh it before applying changes.

## This repository

| Hostname | Settings to review when changing the domain |
|---|---|
| `feedback-agent.xuananh1.site` | README.md hosted-demo links and BASE examples; client URLs |

The audited deployment uses plain HTTP on port 8898. deploy/Caddyfile listens on :8898 and forwards to app:8000 without matching a hostname. Keeping the same server and port requires DNS and public/client link updates; a domain change alone does not require rewriting that Caddyfile. Preserve deployment secrets and the state volume.

## Migration and maintenance

Keeping the domain while changing registrar generally preserves application URLs when the authoritative DNS provider and nameservers remain unchanged. Buying a different domain requires reviewing DNS, tunnel routes, application settings, connectors and client links.

The Dropbox bundle includes `inventory.json`, `dependencies.json` and `prepare-domain.py`. Run the script with the new apex domain to generate a plan for review; it does not apply changes. Resolve all listed blockers before deployment.

After changing a hostname or connector, update the Dropbox inventory and this repository's links. Keep credentials in the existing secret store; the shared recovery documents contain no token or password values.
