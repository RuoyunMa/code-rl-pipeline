# Per-repo training — design doc + industry survey

**Status:** design + industry survey. Implementation queued for Block E3+ in `plan_v3.md`.

## 1 · Problem statement

Modern IDE-assistant products (Cursor, Copilot, Sourcegraph Cody, Continue.dev,
Tabnine, Cody, Trae) **must produce code that fits a specific user codebase** —
matching its API surface, naming conventions, internal idioms, and avoiding
hallucinated cross-file references. The current industry default is **retrieval +
long-context** (RAG over the repo, dump relevant files into the prompt). This
works moderately well but has known failure modes:

- Context length blow-up (often 50K+ tokens just for context, slow + expensive)
- Retrieval miss → model hallucinates symbols that don't exist
- The base model still has zero **muscle memory** for repo idioms; it has to
  re-learn from context every single request
- Fine-grained code style (e.g. "use our `logger.info` not `print`") is
  context-noise-prone

**Per-repo fine-tuning** — training a small LoRA adapter on the codebase itself
— is an under-explored alternative that bakes repo-specific behavior into the
weights. The model "knows" the codebase in a way that RAG + context cannot fully
substitute.

## 2 · Industry survey

| Tool | Per-repo strategy (as of 2025-2026) |
|---|---|
| Cursor Composer 2 | RAG + long context. Composer 2 base is Kimi K2.5 + Cursor's continued pretraining on coding corpus, but NOT per-customer FT. Their internal CursorBench measures real-codebase tasks — they care about this metric, just not via FT path. |
| GitHub Copilot | `@workspace` (RAG over current workspace) + base model. No per-repo FT. |
| Sourcegraph Cody | "Code graph" — structural retrieval over symbols, calls, imports. Multi-repo aware. Still RAG, no FT. |
| Tabnine | Most pertinent to us — "**project training**" tier offers per-project model adaptation. Closest to per-repo LoRA in spirit. Available only to enterprise tier; mechanics not public. |
| Continue.dev | Open-source IDE plugin. Pure RAG. |
| Aider | RAG via repo-map (symbol summarization). No FT. |

**Gap:** open-source / public research has very few examples of per-repo FT pipelines being shown to work materially better than RAG. The closest published works:
- "RepoFusion" (Shrivastava et al. 2023) — uses repo context but at retrieval time
- "RepoCoder" — iterative retrieval, not fine-tuning
- "RepoHyper" — graph-augmented prompting

**Why under-explored**:
1. Most products want one-model-fits-all (multi-tenant serving)
2. LoRA adapter hot-swap in production inference servers is relatively new
3. Per-repo data quality is the bottleneck (not enough samples per repo without synthetic generation)
4. Compute cost (per-customer fine-tuning seemed expensive — though LoRA changes this)

**Production angle:** organizations with large internal monorepos and consistent code-review history have unique data leverage. A per-repo fine-tuning workflow that runs as a CI job on push gives every team an adapter tracked to HEAD. This is genuinely differentiated infra-side work, not a Composer 2 / Copilot feature parity.

## 3 · Hypothesis & success criteria

> **H1**: A 5-20 MB LoRA r=16-32 adapter trained on ~2-5k FIM samples derived
> from a target repo materially improves next-token / fill-in-middle completion
> quality on held-out chunks of the same repo, while preserving generic coding
> ability (HumanEval / MBPP / LiveCodeBench within 2pp of pre-adapter baseline).

Measurable:
- **Same-repo FIM held-out**: edit-similarity (CodeBLEU or token edit-ratio), exact-match middle, symbol-name recall (does the adapter use repo's actual identifiers vs hallucinated ones?)
- **Generic regression check**: LiveCodeBench v6 pass@1 with/without adapter — drop must be ≤2pp
- **Subjective**: 10 cherry-picked qualitative examples reviewed side-by-side

> **H2**: A single base model + N hot-swappable LoRA adapters serves N codebases
> with per-request adapter switching at <100ms overhead via vLLM 0.21
> `LoRARequest` API.

Measurable:
- Latency: TTFT for adapter-cold-load vs hot-cache
- Memory: max adapters simultaneously resident in vLLM
- Throughput: requests/sec mixing different adapters

> **H3 (stretch)**: Per-repo RL with verifiable reward = repo's own pytest passes
> gives further measurable gains over the FIM-SFT-only adapter on
> functional-correctness tasks within the repo.

## 4 · Experimental design

### Target repos (5090 fits these comfortably)

Selection criteria: (a) mature OSS Python repo with mature test suite, (b)
non-trivial size (>10K LoC) so per-repo idioms exist, (c) high test coverage
so we can use pytest as a verifiable reward.

Picks:

| # | Repo | LoC | Tests | Why |
|---|---|---:|---|---|
| 1 | `pallets/flask` | ~16K | comprehensive | Mature, idiomatic, good docstrings |
| 2 | `psf/requests` | ~10K | comprehensive | Smaller, faster turnaround |
| 3 | `pydantic/pydantic` | ~50K | extensive | More complex; tests cover behavior |
| 4 | (custom) a the user-controlled small repo | ~5K | wide | Sanity / data quality check |

Start with Flask. Each repo run takes ~3h end-to-end (FIM gen, train, eval, demo).

### Data construction (FIM samples)

Qwen2.5-Coder tokenizer has explicit FIM tokens:
- `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`

Pipeline:
1. `git clone` target repo @ specific commit (record SHA in meta.json)
2. Filter to `.py` files; exclude `tests/`, `docs/`, `examples/`, `*.pyc`, generated files
3. For each remaining file (~200-1000 files):
   - Tokenize with Qwen tokenizer
   - Drop if <100 tokens or >8000 tokens
   - Sample 5-10 random `(prefix_len, middle_len, suffix_len)` splits with
     middle_len ∈ [20, 200] tokens, prefix+suffix ≤ 2000 tokens
   - Construct FIM training sample: `f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"`
4. **Held-out**: last 10% of files (or last commit's modified files) → eval set
5. Result: ~1k-5k train samples, ~200-500 eval samples per repo

### Training

Base: `outputs/dpo_v2_merged` (our best general-purpose checkpoint from D2).

```
python scripts/train_per_repo_lora.py \
    --base outputs/dpo_v2_merged \
    --data data/per_repo/flask/train.jsonl \
    --output-dir outputs/per_repo/flask_r16 \
    --lora-r 16 --lora-alpha 32 \
    --epochs 2 \
    --lr 1e-5      # lower than general SFT — narrow data
```

Memory budget on 1.5B base (5090 32 GB) — should fit comfortably:
- ~12 GB peak (same as our 1.5B LoRA SFT)
- For 7B base when we move up: ~26 GB (tighter)

Time: ~5-10 min for 1-5k samples × 2 epochs on 1.5B.

**Pushing 5090 to limits** (per the user's ask): train multiple repo adapters concurrently in
the same process via gradient accumulation across mixed-repo batches. Or run a single
heavy variant: per-repo r=64, 5 epochs, with 7B base. ~30-45 min per repo at 7B.

### Eval

#### Same-repo FIM (primary)

For each held-out FIM sample:
- Run model with adapter to fill `<|fim_middle|>` slot
- Compute:
  - **Exact-match** on middle (strict)
  - **Token edit-similarity** (Levenshtein at token-level, normalized 0-1)
  - **Symbol recall**: extract identifiers from gold + predicted middle; recall of repo-internal symbols
  - **Compile rate**: does (prefix + predicted_middle + suffix) parse as valid Python?

Compare:
- (a) Base model (no adapter)
- (b) Base + adapter
- (c) Base + RAG (closest 5 chunks from same repo in context) — to compare FT vs RAG

#### Generic regression (must-not-degrade)

- HumanEval pass@1
- MBPP pass@1
- LiveCodeBench v6 pass@1

Tolerance: ≤2pp drop. If adapter degrades general ability more than that, the LoRA r is too large or training is too aggressive.

#### vLLM hot-swap demo

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="outputs/dpo_v2_merged", enable_lora=True, max_lora_rank=64)
# Multiple adapters live simultaneously, swap per-request
flask_lora = LoRARequest("flask", 1, "outputs/per_repo/flask_r16")
requests_lora = LoRARequest("requests", 2, "outputs/per_repo/requests_r16")
out = llm.generate([prompt1, prompt2],
                   sampling_params,
                   lora_request=[flask_lora, requests_lora])
```

Measure:
- Adapter cold-load latency (first request after `LoRARequest` created)
- Hot-cache latency (subsequent requests using same adapter)
- Throughput when mixing requests across adapters

### Stretch: per-repo RL

Reward signal: for code-completion-of-a-function tasks, can we use the repo's own pytest as oracle?

```
For each held-out function in target repo:
    Mask the function body
    Generate replacement via model + adapter (T=0.8, n=4)
    Replace in-repo, run pytest tests/*test_<this_function_indirectly>*.py
    Reward = 1 if all related tests pass, -1 otherwise
```

Same `train_grpo_trl.py` framework as our D8 work. Per-repo RL = `train_grpo_trl.py` with a `code_reward_per_repo.py` reward fn that:
1. Writes the candidate code into the cloned repo
2. Runs the repo's pytest subset
3. Returns binary reward

Memory + compute: pytest subprocess per candidate × num_gen × per_step → very expensive (could be 30-60 sec/step). For Flask, ~200 steps would be 1-2 hours.

## 5 · Sequence of work (mapped to plan_v3 blocks)

| Block | Sub-work | Time |
|---|---|---:|
| E3a | Build per-repo data pipeline (`fetch_repo.py`, `build_fim_samples.py`) for Flask | 30 min |
| E3b | Train Flask LoRA r=16 on `dpo_v2_merged` (1.5B) | 15 min |
| E3c | Eval Flask LoRA: held-out FIM, regression on HE/MBPP/LCB | 30 min |
| E3d | LoRA hot-swap demo: load Flask + Requests adapters, measure swap latency | 30 min |
| E3e | (stretch) repeat for Requests + Pydantic | 1.5h |
| E3f | (stretch v2) per-repo GRPO on Flask with pytest reward | 2h |
| E3g | Write `results/per_repo_flask.md` with table + qualitative examples | 30 min |

**Total E3:** 3-4h for core (E3a-d), 5-7h with stretches.

## 6 · Why this is worth pursuing

1. **Direct product alignment**: this is what an IDE assistant does at inference time. No company is publicly demonstrating it via FT — opportunity to be first.
2. **Combines everything else in this repo**: LoRA, vLLM serving, eval rigor, verifiable RL — composed for a real product feature.
3. **Adapter-swap serving infra** is non-trivial: multi-tenant inference for many codebases on one base model is a real systems problem.
4. **Honest framing**: "RAG is necessary but not sufficient; here's what FT adds." Complementary to context retrieval, not a replacement.

## 7 · Decision points

Before launching E3:
- [ ] Confirm Flask + 1 sanity-check repo as Phase 1 scope (vs 3-5 repos right away)
- [ ] LoRA r choice: r=16 (small, hot-swap-friendly) vs r=64 (more capacity, harder to swap many at once)
- [ ] Whether to run on top of `dpo_v2_merged` (1.5B, fast iteration) OR 7B (slower, more interesting numbers)
- [ ] Whether to attempt the per-repo RL stretch in this session or defer

I default-recommend: Flask only, r=16, on `dpo_v2_merged` (1.5B), defer RL stretch.
Cheap, fast, gets the demo + numbers in 3h. Repeat for Requests if time.

## Sources

- [Magicoder paper](https://arxiv.org/abs/2312.02120) — for FIM-style training discussion
- [vLLM LoRA serving docs](https://docs.vllm.ai/) — `LoRARequest` API
- [Cursor's CursorBench post](https://cursor.com/blog/cursorbench) — repo-task evaluation
- [Sourcegraph Cody architecture](https://sourcegraph.com/blog/cody-architecture) — RAG approach
- Tabnine project training — enterprise-only, not publicly documented
