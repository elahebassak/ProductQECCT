#!/bin/bash
# One-time environment setup on the Nibi login node (needs internet for ldpc / pymatching).
#   bash slurm/setup_env_nibi.sh
set -e
module load python/3.11
ENV=${PQECCT_ENV:-$HOME/pqecct_env}
if [ ! -d "$ENV" ]; then
  python -m venv "$ENV"
fi
source "$ENV/bin/activate"
pip install --no-index --upgrade pip
# Alliance wheelhouse (fast, no internet needed)
pip install --no-index torch numpy scipy
# classical baselines (PyPI; the login node has internet access)
pip install pymatching stim sinter tqdm
# ldpc: the PyPI source tarballs are missing src_cpp/rng.hpp and Alliance Python rejects manylinux
# wheels, so build from the GitHub repository (has all headers; needs g++ and a few minutes)
pip install "git+https://github.com/quantumgizmos/ldpc.git@v2.4.1" || pip install "git+https://github.com/quantumgizmos/ldpc.git"
python - <<'EOF'
import torch, numpy, scipy, pymatching, ldpc
from ldpc import BpOsdDecoder, BpDecoder
print("torch", torch.__version__, "cuda build:", torch.version.cuda)
print("pymatching", pymatching.__version__, "ldpc", ldpc.__version__)
EOF
echo "Environment ready: $ENV"
