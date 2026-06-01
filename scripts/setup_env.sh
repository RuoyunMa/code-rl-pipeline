#!/usr/bin/env bash
# Bootstrap the `coderl` conda env on RTX 5090 (Blackwell sm_120 / CUDA 13).
# This env runs: vLLM 0.21+, transformers 5.8+, eval scripts, AWQ quantization.
#
# **Not for SFT/DPO/GRPO training** — those go into separate envs:
#   - `coderl_sft` for unsloth + trl (see scripts/setup_env_sft.sh)
#   - `coderl_verl` is the verl Docker image, not a conda env
#
# Why separate envs (verified 2026-05-15 D1, then again D2):
#   - unsloth_zoo pins torch==2.10, which breaks vllm 0.21's C ABI
#   - sglang installs flash-attn-4 4.0.x.bN, which collides with vllm's
#     classic flash_attn import (no .ops submodule)
#
# Target stack (locked as of 2026-05-19):
#   - python 3.11 (conda-forge)
#   - torch 2.11.0 (pypi default — pypi NOW ships +cu130 by default; the old
#     `--index-url cu129` in v1 of this script gave us +cu129 which was the
#     root cause of a multi-hour debugging session, see results/D1_journal.md)
#   - vllm 0.21.0
#   - transformers 5.8.1, datasets 4.8.5, peft 0.19+, accelerate 0.34+
#   - bitsandbytes 0.49.2
#   - cuda-toolkit 13.0 from conda-forge (provides nvcc + dev headers for
#     flashinfer JIT compilation on sm_120 — vllm 0.21's sampling needs this)

# NOTE: not using `set -u` — conda-forge's cuda-nvcc activate hook references
# NVCC_PREPEND_FLAGS without initialization. Keep `-e` and `pipefail`.
set -eo pipefail

ENV_NAME="${ENV_NAME:-coderl}"
PY_VER="3.11"

log() { echo -e "\033[1;36m==> $*\033[0m"; }

# --- 0. Sanity: GPU + driver ---
log "[0/12] Checking nvidia-smi / driver"
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found." >&2
    exit 1
fi
nvidia-smi | head -n 20

# --- 1. Miniconda check + conda env ---
log "[1/12] Creating conda env '$ENV_NAME' (python $PY_VER, conda-forge only)"
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:" >&2
    echo "  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" >&2
    echo "  bash /tmp/miniconda.sh -b -p \$HOME/miniconda3 && \$HOME/miniconda3/bin/conda init bash" >&2
    exit 1
fi

# Ensure conda-forge is the default channel (avoids Anaconda commercial ToS).
conda config --set channel_priority strict
conda config --add channels conda-forge 2>/dev/null || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    env exists, reusing"
else
    conda create -n "$ENV_NAME" -c conda-forge --override-channels python="$PY_VER" -y
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# --- 2. pip into the env (conda-forge python doesn't ship pip) ---
log "[2/12] Ensuring pip is in the env"
if ! command -v pip &>/dev/null; then
    conda install -n "$ENV_NAME" -c conda-forge --override-channels pip -y
fi
pip install --upgrade pip wheel setuptools

# --- 3. CUDA toolkit + host compiler (needed for flashinfer JIT) ---
log "[3/12] Installing cuda-toolkit 13.0 + gxx_linux-64 (host compiler for nvcc)"
# cuda-nvcc 13.0.88 matches the cuda-runtime version that torch 2.11+cu130 bundles.
# gxx_linux-64 provides `x86_64-conda-linux-gnu-cc` which flashinfer's JIT calls via -ccbin.
conda install -n "$ENV_NAME" -c conda-forge --override-channels \
    cuda-toolkit=13.0 gxx_linux-64 -y

# --- 4. PyTorch ---
log "[4/12] Installing PyTorch (pypi default = cu130 wheel in 2026)"
# DO NOT use --index-url cu129 — pypi default is the correct cu130 build.
# This is the variant that pairs with vllm 0.21.0's libcudart.so.13 expectations.
pip install --upgrade torch==2.11.0 torchvision==0.26.0

# --- 5. HF stack + vllm-compat pins ---
log "[5/12] Installing HF stack + vllm structured-output deps"
# vllm 0.21.0 requires these versions specifically (looser pins cause downgrades
# at sglang/other install time):
#   xgrammar>=0.2.0, outlines_core==0.2.14, llguidance>=1.3,<1.4
pip install --upgrade \
    "transformers>=5.8,<5.9" \
    "accelerate>=0.34" \
    "datasets>=4.0" \
    "peft>=0.14" \
    sentencepiece protobuf safetensors \
    "xgrammar>=0.2.0" \
    "outlines_core==0.2.14" \
    "llguidance>=1.3.0,<1.4.0"

# --- 6. bitsandbytes (Blackwell sm_120 kernels) ---
log "[6/12] Installing bitsandbytes >= 0.46"
pip install "bitsandbytes>=0.49"

# --- 7. vLLM ---
log "[7/12] Installing vLLM 0.21"
pip install "vllm>=0.21,<0.22"

# --- 8. Quant + tracking + utils ---
log "[8/12] Installing AutoAWQ + W&B + utils"
pip install \
    "autoawq>=0.2.7" \
    wandb \
    tqdm pyyaml jsonlines pandas numpy pyarrow

# --- 9. Symlink CUDA headers / libs into conventional locations ---
# conda-forge cuda-toolkit puts headers in $CONDA_PREFIX/targets/x86_64-linux/include/
# and libs in .../lib/, but flashinfer's build.ninja expects them in
# $CONDA_PREFIX/include and $CONDA_PREFIX/lib64. Symlinking is faster than rebuild.
log "[9/12] Symlinking CUDA headers (include/) + libs (lib64/) + stubs"
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
# libcuda.so stub for link-time -lcuda resolution
STUB=$(find "$CONDA_PREFIX" -path "*/stubs/libcuda.so" 2>/dev/null | head -1)
if [ -n "$STUB" ]; then
    ln -sf "$STUB" "$CONDA_PREFIX/lib64/stubs/libcuda.so"
fi

# --- 10. Remove any leftover flash_attn namespace directory ---
# A stale empty `flash_attn` dir (e.g. from a prior flash-attn-4 install/remove)
# makes `importlib.util.find_spec("flash_attn")` return non-None, which causes
# vllm 0.21's rotary embedding init to attempt importing `.ops` and crash.
log "[10/12] Removing stale flash_attn namespace dir if present"
SP="$CONDA_PREFIX/lib/python$PY_VER/site-packages"
if [ -d "$SP/flash_attn" ] && [ -z "$(ls -A "$SP/flash_attn" 2>/dev/null)" ]; then
    rmdir "$SP/flash_attn"
    echo "    removed empty $SP/flash_attn"
fi

# --- 11. Patch transformers PACKAGE_DISTRIBUTION_MAPPING KeyError ---
# transformers 5.8.1 dereferences PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]
# directly. When no flash-attn package is installed and the key is absent,
# this raises KeyError during `import transformers.models.auto.image_processing_auto`,
# which vllm pulls in during LLM() construction.
# Patch to use .get(..., []) — bug confirmed and reported upstream.
log "[11/12] Patching transformers KeyError for missing flash_attn key"
TPATCH="$SP/transformers/utils/import_utils.py"
if [ -f "$TPATCH" ] && grep -q 'PACKAGE_DISTRIBUTION_MAPPING\["flash_attn"\]' "$TPATCH"; then
    cp -n "$TPATCH" "$TPATCH.orig" || true
    sed -i 's/PACKAGE_DISTRIBUTION_MAPPING\["flash_attn"\]/PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn", [])/g' "$TPATCH"
    sed -i 's/PACKAGE_DISTRIBUTION_MAPPING\["flash_attn_interface"\]/PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn_interface", [])/g' "$TPATCH"
    echo "    patched $TPATCH"
fi

# --- 12. Env-activate hooks: LD_LIBRARY_PATH, CUDA_HOME, allocator config ---
log "[12/12] Writing conda activate hooks"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_alloc.sh" <<'ENVSH'
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENVSH
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_home.sh" <<'ENVSH'
export CUDA_HOME="$CONDA_PREFIX"
ENVSH
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_ld_path.sh" <<'HOOK'
# Add bundled nvidia-*-cu13 lib dirs to LD_LIBRARY_PATH so vllm._C and torch
# can find libcudart.so.13, libcudnn.so.9, libnccl.so.2, libcusparseLt.so.0 etc.
_NVIDIA_LIB_DIRS=$(python -c "
import os, glob, sys
sp = os.path.join(sys.prefix, 'lib', 'python3.11', 'site-packages')
dirs = [d for d in glob.glob(sp+'/nvidia/*/lib') if os.listdir(d)]
print(':'.join(sorted(dirs)))
" 2>/dev/null)
if [ -n "$_NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="$_NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _NVIDIA_LIB_DIRS
HOOK

# Re-source so the rest of this script uses the new LD_LIBRARY_PATH
source "$CONDA_PREFIX/etc/conda/activate.d/cuda_ld_path.sh"
source "$CONDA_PREFIX/etc/conda/activate.d/cuda_home.sh"
source "$CONDA_PREFIX/etc/conda/activate.d/cuda_alloc.sh"

# --- Verify ---
log "Verifying CUDA + Blackwell capability + key imports"
python - <<'PY'
import torch, transformers, vllm, datasets, peft, bitsandbytes as bnb
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"vllm          {vllm.__version__}")
print(f"datasets      {datasets.__version__}")
print(f"peft          {peft.__version__}")
print(f"bitsandbytes  {bnb.__version__}")
print(f"cuda avail    {torch.cuda.is_available()}")
print(f"cuda runtime  {torch.version.cuda}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device        {p.name}")
    print(f"capability    {cap}   {'OK (Blackwell)' if cap == (12, 0) else 'WARN: expected (12, 0)'}")
    print(f"vram total    {p.total_memory / 1e9:.1f} GB")
# Confirm vllm can actually construct LLMEngine (catches flash_attn / KeyError regressions)
from vllm import LLM, SamplingParams
print("vllm LLM symbol import OK")
PY

cat <<EOF

────────────────────────────────────────
Done. Activate with:
    conda activate $ENV_NAME

Env-activate hooks auto-set:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_HOME=\$CONDA_PREFIX
    LD_LIBRARY_PATH includes nvidia/*/lib for libcudart.so.13 etc.

For SFT (unsloth + trl in torch 2.10 land), use a separate env:
    bash scripts/setup_env_sft.sh

For verl GRPO, use the official Blackwell Docker image (not this conda env).
────────────────────────────────────────
EOF
