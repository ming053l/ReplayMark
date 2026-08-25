# Paper experiments

This directory retains only the experiment and plotting scripts associated with the reported
paper results. The corresponding measured outputs are committed under `results/`.

The ReplayMark scripts use the model and dataset configuration described in the repository
README. Peer-method scripts additionally require the authors' dgMARK implementation checked out
under `baselines/dgmark-watermarking`; that third-party repository is not redistributed here.
Paths stored in these files document the original runs. For new generation and verification, use
the portable package API or `examples/quickstart.py`.
