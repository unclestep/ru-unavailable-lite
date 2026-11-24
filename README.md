# rustrict

Automated weekly pipeline that builds clean, compact split-routing lists for Xray (`geosite.dat`, `geoip.dat`, `split-rules.json`).

## Release Artifacts

- `split-rules.json` — deduplicated `domain:<host>` rules and collapsed service CIDRs (excluding `PRIVATE` and shared CDNs) ready to embed directly into client JSON profiles without external `.dat` files.
- `geosite.dat` — protobuf geosite database with `RU-BLOCKED` and `RU-FORBIDDEN` categories.
- `geoip.dat` — protobuf geoip database containing `PRIVATE`, `TELEGRAM`, `FACEBOOK`, `TWITTER`, `NETFLIX`, and first-party `GOOGLE` infrastructure (`goog.json` minus GCP tenant `cloud.json`).
- `ru-blocked.txt`, `ru-forbidden.txt`, `stats.json`, `checksums.sha256sum`.

## Pipeline Overview

1. **Sources:** Fetches the RKN registry (`ru-blocked`) and community-curated geoblock lists (`ru-forbidden`).
2. **Normalization & Farm Removal:** Applies the Public Suffix List, removes subdomains covered by parent entries, and drops multi-subdomain spam/doorway farms (e.g., `*.orgs.biz`, piracy doorway farms).
3. **CrUX Popularity Filter (`ru-blocked` only):** Filters against the latest RU Chrome User Experience Report snapshot (apex domains matched by registrable root, subdomains matched by exact host).
4. **Single-Pass DNS & GeoIP:** Resolves `A`/`AAAA` once and maps IPv4/IPv6 against `sapics/ip-location-db`. Drops dead and RU-hosted domains from `ru-blocked`, and drops only stray RU-hosted entries from `ru-forbidden`.
5. **Deterministic Filtering (`ru-blocked` only):** Protects `allowlist.txt` and `ru-forbidden`, then filters against UT1, StevenBlack (porn + gambling), HaGeZi gambling, AdGuard phishing, a mirror-cluster detector, and strict RU junk rules.
6. **SSRF-Safe HTML Probe:** Asynchronously fetches landing-page metadata (`<title>`, `<meta>`, `<h1>`, text snippet) with a gzipped cache. Enforces public-IP-only DNS and IP-literal checks on every redirect hop (max 3), 64 KB body cap, `text/html`-only reads, no cookies/JS, and scrubbed CI credentials.
7. **Two-Stage ML Classifier:** Combines HTML+Domain rules and distant-supervision content classification with a Domain+Structural fallback (`char_wb` TF-IDF + morphology/TLD/cluster features) for `403`/bot-challenge pages.
8. **Export & Verification:** Encodes and round-trip verifies `geosite.dat`, `geoip.dat`, `split-rules.json`, and `checksums.sha256sum`.

## Local Usage

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m rustrict.build --stage all
```
