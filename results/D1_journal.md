# D1 Journal — 2026-05-15

**Status:** D1 acceptance ✅ (baseline 跑通 + 数据记录) · M1 部分完成（baseline ✅ · repo/Docker/verl 推到 D2）

---

## 最终数字

| Benchmark | pass@1 | n_pass / total | T | n |
|---|---:|:-:|:-:|:-:|
| HumanEval | **73.17%** | 120/164 | 0.0 | 1 |
| MBPP (sanitized) | **63.42%** | 163/257 | 0.0 | 1 |

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Backend: vLLM 0.21.0 · dtype=bf16 · gpu_mem_util=0.85
- 文件: `results/baseline.md`, `results/humaneval_baseline.{jsonl,summary.json}`, `results/mbpp_baseline.{jsonl,summary.json}`

> ⚠️ **HumanEval 73.17% 比 CLAUDE.md 写的 ~62% 高 11pp**。Qwen2.5-Coder-1.5B-Instruct 官方 release page 报的也是 ~70.8%（T=0 greedy），我们 73 跟官方对得上，可信。
>
> 后果：M2/SFT 的 acceptance "+3-6% over base" 要变成 "+1-3%"（76-79% 接近 7B 量级天花板，1.5B SFT 单步很难拉这么大）。同样 M4 (DPO) "+1-3% over SFT"、M5 (GRPO) "≥ DPO eval" 也要相应缓和。**D4 evaluating 前必须重订阈值，否则会按假目标判定失败**。

---

## 5090 当前环境状态

### 通了的（D1 收尾时验证过）

- SSH 永久免密：`~/.ssh/id_ed25519` → 5090 的 `authorized_keys`，SSH config `Host 5090`
- Miniconda：`~/miniconda3`, conda 26.3.2, channels = conda-forge only（去 ToS）
- conda env **coderl**：
  - python 3.11.15
  - **torch 2.11.0+cu130**（注意：不是 cu129！pypi 默认 wheel 已经是 cu130）
  - vllm 0.21.0
  - bitsandbytes 0.49.2
  - cuda-toolkit 13.0 (conda-forge) — nvcc + dev headers
  - 全套 nvidia-*-cu13 wheels（cudnn 9.22, nccl 2.30.4, cusparselt 0.9.1, nvshmem 3.6.5 ……）
- `~/workspace/code-rl-pipeline/`（已从 `~/code-rl-pipeline` 搬过来；本地 `deploy.sh` 默认 REMOTE_DIR 已更新）

### 已知坏的（D2 AM 必须先修）

1. **`coderl` env 被 sglang 安装搅了一半**（晚上 sglang 尝试同 env 共存时副作用）：
   - transformers 5.8.1 → 5.6.0
   - xgrammar 0.2.0 → 0.1.32
   - llguidance 1.3.x → 0.7.30
   - outlines_core 0.2.14 → 0.1.26
   - nvidia-cudnn-cu13 9.22.0.52 → 9.19.0.56
   - nvidia-cusparselt-cu13 0.9.1 → 0.8.0
   - nvidia-nccl-cu13 2.30.4 → 2.28.9
   - nvidia-nvshmem-cu13 3.6.5 → 3.4.5
   - 装了 `flash-attn-4` 4.0.0b13（占了 `flash_attn` namespace 但没 `.ops` 子模块，跟 vllm 冲突），我已手动 uninstall → vllm 当前 import 也挂（transformers 5.6 `PACKAGE_DISTRIBUTION_MAPPING['flash_attn']` KeyError）
   - snapshot 在 `~/workspace/code-rl-pipeline/logs/env_before_sglang.txt`（pip freeze 200 行）
2. **5090 系统级 suspend 没禁用**：晚上挂起了 → SSH 断 → 阻断 D1 收尾。D2 AM 第一件事是 wake + 禁 suspend。
3. **W&B 没登录**：D1 baseline 用 NO_WANDB=1。D2 一行 `wandb login` 解决。
4. **Docker 没装**：D2 EVE verl Docker 要用，得 `apt install docker.io` + `nvidia-container-toolkit`。
5. **Unsloth / Unsloth_zoo / trl 已卸**：D3 SFT 前要解决 unsloth_zoo 把 torch 拽回 2.10 的 dep pin 问题，否则就改用 `trl.SFTTrainer`（更稳但慢）。

---

## D2 启动清单（按依赖顺序）

```
[1] Wake 5090（敲键盘）
[2] 禁 suspend:
    sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
    gsettings set org.gnome.desktop.session idle-delay 0
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
[3] 修 coderl env（精确回滚 vllm 所需依赖）:
    pip install --force-reinstall transformers==5.8.1 xgrammar==0.2.0 outlines_core==0.2.14 \
        "llguidance>=1.3.0,<1.4.0"
    pip install --force-reinstall --no-deps \
        nvidia-cudnn-cu13==9.22.0.52 nvidia-cusparselt-cu13==0.9.1 \
        nvidia-nccl-cu13==2.30.4 nvidia-nvshmem-cu13==3.6.5
    # 验证: python -c "from vllm import LLM, SamplingParams; print('OK')"
    # 如果 transformers KeyError: 'flash_attn' 还在 → pip install flash-attn 但要预编译 wheel
[4] wandb login → 输 API key
[5] 装 docker:
    sudo apt install -y docker.io
    sudo usermod -aG docker $USER
    # 装 nvidia-container-toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    # newgrp docker 或重 ssh
[6] 拉 verl Blackwell Docker → 跑 1-step GRPO sanity
[7] sglang 对比：起独立 env coderl_sglang（不要再同 env 装！）
[8] M1 acceptance check
```

---

## 设计决策（D1 期间确认）

1. **setup_env.sh 的 `--index-url cu129` 是个 bug**：pypi 默认 torch 在 2026 已经是 cu130 build。下次新机器 reproduce 要么：
   - 删掉 `--index-url`，让 pip 走 pypi 默认（cu130）
   - 或改成 `cu130`
   - 临时 patch 没做，因为 5090 这台已经手动修过了
2. **sglang 同 env 已确认不可行**：flash-attn-4（sglang 用）和 classic flash_attn（vllm 用）共占 `flash_attn` namespace。装 classic flash-attn 还要从源码编译 nvcc（20-30 min）。D2 起独立 env coderl_sglang，5GB 多点磁盘代价换隔离稳定。
3. **vllm 0.21 在 sm_120 (Blackwell) 上的最小可用栈**：
   - torch 2.11+cu130（pypi 默认）
   - cuda-toolkit 13.0 (conda) — nvcc + dev headers
   - 把 `$CONDA_PREFIX/targets/x86_64-linux/include/*` 软链到 `$CONDA_PREFIX/include/`
   - 把 `$CONDA_PREFIX/targets/x86_64-linux/lib/{libcudart.so*, stubs/libcuda.so}` 软链到 `$CONDA_PREFIX/lib64/`
   - flashinfer JIT 用 conda 的 `x86_64-conda-linux-gnu-cc` 作 host compiler
   - 启动时需要 `LD_LIBRARY_PATH=` 覆盖 nvidia/*/lib —— 已写进 `$CONDA_PREFIX/etc/conda/activate.d/cuda_ld_path.sh` 自动激活
4. **vllm 选 FLASH_ATTN 后端（不是 flashinfer）**：从 logs/run_baseline.log 看 `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` —— sm_120 + bf16 + 1.5B 默认走 FLASH_ATTN，不需要触发 flashinfer JIT 编译完整 attention kernel（但 flashinfer 还是用于 sampling top-p/top-k）。

---

## sglang 已收集数据（如果决定后续对比）

- bs=1, in=256, out=256: prefill 36710 tok/s, decode median 347.57 tok/s, total 691.82 tok/s
- bs=8, in=256, out=256: prefill 63300 tok/s, decode median 2473 tok/s, total 4775 tok/s
- bs=32, in=256, out=256: prefill 71834 tok/s, decode median 8844 tok/s, total 15788 tok/s
- 文件：`logs/bench/sglang.log`
- **vllm 对应数据没拿到**（被 flash_attn 冲突拦下），D2 修好 coderl 后跑 `scripts/bench_vllm.py` 同形状对比即可。

---

## 时间消耗记录

实际 D1 时间 ~3-4 小时，主要消耗：
- env 折腾（cu129/cu130 mismatch + 各种 nvidia 包没解压完整 + cuda-nvcc + flashinfer JIT 缺 headers/stubs）：~1.5 h
- baseline 跑通后实际推理：**2 分钟**（vLLM @ 5090 + 1.5B + greedy 快到爆）
- sglang 尝试 + 撞同 env 冲突：~30 min
- 5090 suspend 断连：blocked

教训：**新机器第一次跑 vllm + Blackwell + CUDA 13 的栈，预留 2 小时 env 折腾时间**，不要按"15-30 min setup"的乐观估计 plan。
