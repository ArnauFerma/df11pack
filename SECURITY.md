# Security

## Reporting

Please report vulnerabilities privately through GitHub:
[**Report a vulnerability**](https://github.com/ArnauFerma/df11pack/security/advisories/new).
Do not open a public issue. You should get a reply within a week; a fix and an
advisory follow once the problem is confirmed.

## Scope

df11pack reads model files you give it and writes files other programs will load.
Most relevant:

- a crafted `.safetensors` or definition file that makes df11pack crash, loop,
  exhaust memory, or read or write outside the paths it was given;
- output that decodes to different weights than the source without `--safe` or
  `verify` noticing.

Loading DF11 files in other programs (DFloat11, ComfyUI) is theirs; report those
upstream.

## Supported versions

Only the latest release receives fixes.
