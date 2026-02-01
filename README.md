# FSU v2 — State-Space + Mixture-of-Experts Language Model
**Author: Pirassena Sabaratnam**

## Overview
FSU v2 is a byte-level language model combining **Gated Linear Attention** (GLA) — a linear-complexity alternative to softmax attention — with **Mixture-of-Experts** (MoE) routing and depthwise causal convolutions. It was developed as a bridge between the continuous field experiments of FSU v1 and the novel physics-based approach of the Gravitational Vector Network (GVN).

## Architecture

### Layer Structure
Each of the 24 layers consists of three components:
1. **Depthwise Causal Conv1D** — Local context aggregation (kernel=4). Captures immediate byte-level dependencies (e.g., multi-byte UTF-8 characters, common bigrams).
2. **Gated Linear Attention** — Global state evolution with data-dependent decay gates. 16 heads with HiPPO-inspired timescale initialization, providing diverse memory lifespans (from ~10 to ~10,000 bytes) across heads.
3. **Mixture of Experts** — 8 SwiGLU expert FFNs with top-2 routing and load-balancing auxiliary loss.

### Key Properties
- **O(1) memory inference**: Recurrent state matrices (one per head per layer) — constant memory regardless of context length
- **Byte-level**: Direct UTF-8 byte processing (vocab=260: 256 bytes + 4 special tokens), no tokenizer
- **Custom Triton kernels**: Fused GLA forward pass for efficient recurrent scan computation
- **Weight-tied embedding**: Shared weights between input embedding and output projection

### Model Configurations
| Config | d_model | Layers | Heads | Experts | ~Params |
|:---|:---:|:---:|:---:|:---:|:---:|
| Small | 768 | 12 | 12 | — | 125M |
| Base | 1024 | 16 | 16 | 4 (top-2) | 350M |
| Large | 2048 | 24 | 16 | 8 (top-2) | 1.3B |
| XL | 4096 | 32 | 32 | 16 (top-4) | 7B |

## Results
FSU v2 produced **coherent English responses** — a significant improvement over the incoherent outputs of FSU v1's PDE-based approach. However, it lacked **long-range context propagation**: responses were grammatically correct but did not sustain topical coherence beyond short spans.

## Why It Was Abandoned
The architecture worked, but it was built entirely from established paradigms (GLA from state-space model literature, MoE from Transformer scaling research). The research goal was not to combine existing methods, but to validate whether the continuous dynamics concept from FSU v1 could work as a **native computational mechanism** — not as a wrapper around known approaches. This led to the development of **GVN**, which replaced attention with N-body gravitational physics and achieved the original research objective.

## License
Apache License 2.0.

---
*Developed January–February 2026 — Auralith Inc.*
