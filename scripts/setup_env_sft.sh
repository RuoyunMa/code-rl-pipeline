#!/usr/bin/env bash
# Bootstrap the `coderl_sft` conda env for SFT (Unsloth + LoRA) training.
#
# Why a separate env from `coderl`:
#   unsloth_zoo pins torch==2.10, which is incompatible with vllm 0.21's
#   C++ ABI (vllm wants torch 2.11). We isolate them.
#
# Workflow:
#   1. Train SFT in this env -> saves LoRA adapter and (optional) merged FP16 model
#   2. Run eval / generation in the `coderl` env using the merged model with vLLM
#
# Target stack (Blackwell sm_120, locked 2026-05-19):
#   - python 3.11 (conda-forge)
#   - torch 2.10 (pulled in by unsloth_zoo)
#   - transformers 5.5+, trl 0.24+, datasets 4.3+
#   - unsloth latest (git), unsloth_zoo
#   - bitsandbytes 0.49+ (Blackwell kernels)

# NOTE: not using `set -u` here — conda-forge's cuda-nvcc activate hook
# references NVCC_PREPEND_FLAGS without initialization, triggering "unbound variable"
# under nounset. Keep `-e` and `pipefail` for actual error catching.
set -eo pipefail

ENV_NAME="${ENV_NAME:-coderl_sft}"
PY_VER="3.11"

log() { echo -e "\033[1;36m==> $*\033[0m"; }

# --- 0. nvidia-smi sanity ---
log "[0/7] Sanity check"
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi missing" >&2; exit 1; }
command -v conda >/dev/null || { echo "ERROR: conda missing — bootstrap Miniconda first" >&2; exit 1; }

# --- 1. Create env ---
log "[1/7] Creating env '$ENV_NAME'"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    env exists, reusing"
else
    conda create -n "$ENV_NAME" -c conda-forge --override-channels python="$PY_VER" -y
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# --- 2. pip + cuda toolkit (for any kernel JIT) ---
log "[2/7] Installing pip + cuda-toolkit + host compiler"
conda install -n "$ENV_NAME" -c conda-forge --override-channels \
    pip cuda-toolkit=13.0 gxx_linux-64 -y
pip install --upgrade pip wheel setuptools

# --- 3. torch first (let unsloth dictate version downstream) ---
# We don't pin torch here — let `unsloth_zoo` pull whatever it needs (torch 2.10).
# Pinning torch first would conflict with unsloth_zoo's pin and cause a yank.

# --- 4. Unsloth + unsloth_zoo (pull in compatible torch/transformers/trl) ---
log "[4/7] Installing Unsloth (Blackwell-aware build) + unsloth_zoo"
# unsloth's pypi package includes unsloth_zoo as a dep; pulls torch 2.10 + transformers 5.5 + trl 0.24
pip install unsloth unsloth_zoo

# --- 5. Companion libs (trl, datasets, bnb already pulled in via unsloth) ---
log "[5/7] Adding companion libs (HF utils, w&b)"
pip install \
    "bitsandbytes>=0.49" \
    "accelerate>=0.34" \
    "peft>=0.14" \
    sentencepiece protobuf safetensors \
    wandb \
    tqdm pyyaml jsonlines pandas numpy pyarrow

# --- 6. Symlink cuda headers/libs (same fix as coderl) ---
log "[6/7] Symlinking CUDA headers/libs/stubs (for any bnb / flash-attn JIT)"
mkdir -p "$CONDA_PREFIX/lib64/stubs"
for f in "$CONDA_PREFIX"/targets/x86_64-linux/include/*.h; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    [ -e "$CONDA_PREFIX/include/$bn" ] || ln -sf "$f" "$CONDA_PREFIX/include/$bn"
done
for d in "$CONDA_PREFIX"/targets/x86_64-linux/include/*/; do
    [ -e "$d" ] || continue
    bn=$(basename "$d")
    [ -e "$CONDA_PREFIX/include/$bn" ] || ln -sf "$d" "$CONDA_PREFIX/include/$bn"
done
for f in "$CONDA_PREFIX"/targets/x86_64-linux/lib/*.so* "$CONDA_PREFIX"/targets/x86_64-linux/lib/*.a; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    [ -e "$CONDA_PREFIX/lib64/$bn" ] || ln -sf "$f" "$CONDA_PREFIX/lib64/$bn"
done
STUB=$(find "$CONDA_PREFIX" -path "*/stubs/libcuda.so" 2>/dev/null | head -1)
[ -n "$STUB" ] && ln -sf "$STUB" "$CONDA_PREFIX/lib64/stubs/libcuda.so"

# --- 7. Env hooks ---
log "[7/7] Writing conda activate hooks"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_alloc.sh" <<'ENVSH'
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENVSH
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_home.sh" <<'ENVSH'
export CUDA_HOME="$CONDA_PREFIX"
ENVSH
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_ld_path.sh" <<'HOOK'
_NVIDIA_LIB_DIRS=$(python -c "
import os, glob, sys
sp = os.path.join(sys.prefix, 'lib', 'python3.11', 'site-packages')
dirs = [d for d in glob.glob(sp+'/nvidia/*/lib') if os.listdir(d)]
print(':'.join(sorted(dirs)))
" 2>/dev/null)
[ -n "$_NVIDIA_LIB_DIRS" ] && export LD_LIBRARY_PATH="$_NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _NVIDIA_LIB_DIRS
HOOK
source "$CONDA_PREFIX/etc/conda/activate.d/cuda_ld_path.sh"
source "$CONDA_PREFIX/etc/conda/activate.d/cuda_home.sh"
source "$CONDA_PREFIX/etc/conda/activate.d/cuda_alloc.sh"

# --- Verify ---
log "Verifying"
python - <<'PY'
import torch, transformers, trl, datasets, peft, bitsandbytes as bnb
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"trl           {trl.__version__}")
print(f"datasets      {datasets.__version__}")
print(f"peft          {peft.__version__}")
print(f"bitsandbytes  {bnb.__version__}")
try:
    import unsloth
    print(f"unsloth       {getattr(unsloth, '__version__', 'unknown')}")
except Exception as e:
    print(f"unsloth       FAIL: {e}")
print(f"cuda avail    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"capability    {torch.cuda.get_device_capability(0)}")
PY

cat <<EOF

────────────────────────────────────────
Done. Activate with:
    conda activate $ENV_NAME

Run SFT:
    python scripts/train_sft.py --output-dir outputs/sft --save-merged

After training, eval in coderl env:
    conda deactivate && conda activate coderl
    python scripts/eval_humaneval.py --model outputs/sft_merged --out results/humaneval_sft.jsonl --no-wandb
────────────────────────────────────────
EOF
