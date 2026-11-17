#!/bin/bash
# One-time environment setup on an Alliance login node (Nibi, Narval, Fir): needs internet for ldpc / pymatching.
#   bash slurm/setup_env.sh
set -e
module load StdEnv/2023 python/3.11 2>/dev/null || module load python/3.11
ENV=${PQECCT_ENV:-$HOME/pqecct_env}
if [ ! -d "$ENV" ]; then
  python -m venv "$ENV"
fi
source "$ENV/bin/activate"
pip install --no-index --upgrade pip
# Alliance wheelhouse (fast, no internet needed); the torch wheel is built for both A100 (Narval) and H100 (Fir/Nibi)
pip install --no-index torch numpy scipy
# classical baselines (PyPI; the login node has internet access)
pip install pymatching stim sinter tqdm
# ldpc: build from the GitHub repository (the PyPI sdist lacks a header and Alliance Python rejects manylinux wheels)
pip install "git+https://github.com/quantumgizmos/ldpc.git@v2.4.1" || pip install "git+https://github.com/quantumgizmos/ldpc.git"
python - <<'EOF'
import torch, numpy, scipy, pymatching, ldpc
from ldpc import BpOsdDecoder, BpDecoder
print("torch", torch.__version__, "cuda build:", torch.version.cuda)
print("pymatching", pymatching.__version__, "ldpc", ldpc.__version__)
EOF
echo "Environment ready: $ENV"
