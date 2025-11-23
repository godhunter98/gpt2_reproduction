# GPT-2 Reproduction

This repo contains a from-scratch PyTorch re-implementation of the GPT-2 architecture together with a lightweight training loop, basic sampling code, and a utility script for evaluating pretrained GPT-2 checkpoints on the HellaSwag benchmark. It is intentionally small, easy to read, and ideal for experimenting with tokenizer pipelines, architecture tweaks, and throughput measurements on your own text corpora.

## Highlights
- End-to-end GPT-2 model definition (multi-head causal self-attention, weight tying, GELU MLP blocks, LayerNorm, configurable depth/width).
- Minimal `DataLoaderLite` that tokenizes a plain text file with `tiktoken` and yields contiguous fixed-length windows for autoregressive training.
- Training loop that supports CPU, CUDA, and Apple Metal (MPS) devices, `torch.compile`, gradient clipping, and AdamW weight-decay grouping.
- Reference inference block that demonstrates top-k sampling to produce short continuations from a prompt.
- `hellaswag.py` script that downloads HellaSwag, renders it into tensors, and measures completion accuracy with Hugging Face GPT-2 checkpoints.

## Project Layout
| Path | Purpose |
| --- | --- |
| `train_gpt2.py` | Primary single-process trainer plus an example sampling routine. |
| `distributed_train_gpt2.py` | Variant of the trainer kept for future distributed experiments. |
| `hellaswag.py` | Downloads and evaluates GPT-2 models on the HellaSwag benchmark. |
| `input.txt` | Sample training corpus (Shakespeare). Replace with your own text. |
| `hellaswag/` | Cache directory for downloaded HellaSwag splits. |
| `play.py` | Tiny helper that inspects the `RANK` env var and tests PyTorch MPS availability. |
| `pyproject.toml` | Minimal project metadata for packaging/linters. |

## Getting Started
1. **Create a Python 3.12 environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install torch tiktoken transformers matplotlib tqdm requests
   ```
   (Install the GPU-enabled PyTorch wheel that matches your CUDA/cuDNN stack if you plan to train on a GPU.)
3. **Prepare data**  
   Replace `input.txt` with any UTF-8 text. `DataLoaderLite` scans the file once, so you can use concatenated books, code, or domain-specific prose. Tokenization happens via the GPT-2 BPE so there is no extra preprocessing step.

## Training
Run the default training loop:
```bash
python train_gpt2.py
```
Key knobs inside the script:
- `GPTconfig`: change `block_size`, number of layers/heads, embedding size, dropout, or vocabulary size when experimenting with different model families.
- `DataLoaderLite(B, T, "input.txt")`: adjust batch size (`B`), sequence length (`T`), or the path to another corpus.
- `num_iters`, `learning_rate`, `weight_decay`: modify the outer-loop hyperparameters near the bottom of the file.

The script detects CUDA automatically and enables `torch.compile(model)` for PyTorch 2.x speedups. Loss curves are appended to `lossi` and plotted with Matplotlib once training finishes. Gradient norms are clipped to 1.0 each step to keep the optimizer stable.

> Note: after the training block the script calls `sys.exit(0)`. Remove/comment out that line if you want to run the example sampling code that follows training.

### Distributed / Multi-Process Experiments
`distributed_train_gpt2.py` mirrors the single-process trainer and is reserved for further distributed data-parallel work. Use it as a starting point if you plan to integrate `torch.distributed` or DeepSpeed—currently it executes the same logic as `train_gpt2.py`.

## Sampling
The lower section of `train_gpt2.py` (executed if you skip the `sys.exit(0)`) demonstrates top-k sampling:
1. Encode an initial prompt with `tiktoken`.
2. Repeatedly append sampled tokens until `max_length` is reached.
3. Decode and print several continuations.

You can also load pretrained checkpoints without training by uncommenting `model = GPT.from_pretrained('gpt2')`.

## Evaluating on HellaSwag
`hellaswag.py` can score any Hugging Face GPT-2 checkpoint on the HellaSwag validation split:
```bash
python hellaswag.py --model_type gpt2-xl --device cuda
```
The script will download the JSONL splits into `hellaswag/`, render each example into tensors, and compute both raw and length-normalized accuracies. Use `--model_type` to pick from `gpt2`, `gpt2-medium`, `gpt2-large`, or `gpt2-xl`. If you want to stop early (e.g., after 100 examples) edit the `if num_total == 100: break` guard at the bottom.

## Troubleshooting & Tips
- **Tokenizer mismatch**: ensure your training text uses UTF-8 and avoid extremely long lines; `block_size` must be >= your desired context window.
- **Memory pressure**: reduce `B`, `T`, or the hidden size (`n_embd`) when running on constrained GPUs/CPUs. Gradient accumulation is easy to add around the existing loop if you need effective larger batches.
- **Matplotlib on headless servers**: set `MPLBACKEND=Agg` if you cannot open GUI windows.
- **HellaSwag download issues**: the script requires outbound network access. Pre-download the JSONL files and drop them into `hellaswag/` if you are on an offline machine.

## Next Steps
- Swap in your own tokenizer or vocabulary for domain-specific experiments.
- Extend `distributed_train_gpt2.py` with `torch.distributed` primitives for multi-GPU scaling.
- Add logging (TensorBoard/W&B) or checkpointing to resume long training jobs.

Happy hacking!
