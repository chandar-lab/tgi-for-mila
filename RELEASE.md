
# Release notes

## Release 2

_October 12th, 2023_

Update to TGI version 1.1.0. This update includes upgrading
flash-attention to version 2.3.2, as well as adding the
EETQ and awq kernels.

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
