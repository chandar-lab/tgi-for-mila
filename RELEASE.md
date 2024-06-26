
# Release notes

## Release 3

_April 11th, 2024_

Update to TGI version 1.4.4, as well as adding `peft` and '`outlines` support,
and including the `exllamav2` and `megablocks` kernels.

On compute canada this also upgrades from `StdEnv/2020` to `StdEnv/2023`.

* **TGI version:** 1.4.4
* **enabled features:** [bnb, accelerate, quantize, peft, outlines]
* **Flash-attention version:** 2.3.2

## Release 2

_October 12th, 2023_

Update to TGI version 1.1.0. This update includes upgrading
flash-attention to version 2.3.2, as well as adding the
EETQ and awq kernels.

In addition, model files are now rsync-copied to the local
filesystem before the TGI server is started. This makes the
TGI startup time faster, and subsequent restarts are
much faster. This change replaces `TMP_PYENV` with `TGI_TMP`.

* **TGI version:** 1.1.0
* **enabled features:** [bnb, accelerate, quantize]
* **Flash-attention version:** 2.3.2

## Release 1

_September 1st, 2023_

The inital release.

Previous versions used a lot of hacks to avoid CUDA 11.8 on Mila.
However, as Mila have now upgraded their drivers that is no longer required.

Previous versions also didn't support the `quantize` feature. `bnb` and
`accelerate` was also not correctly installed.

* **TGI version:** 1.0.2
* **enabled features:** [bnb, accelerate, quantize]
* **Flash-attention version:** 2.0.8
