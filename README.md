# Project for: Principles of Large-Scale Machine Learning @ Cornell 

Each part builds a core ML system from the ground up, progressing from scalar autodiff to neural network training to LLM inference.

---

## Part 1 — Automatic Differentiation Engine

A reverse-mode automatic differentiation engine built entirely in NumPy, without any ML frameworks. The core abstraction is `BackproppableArray`, which wraps a NumPy array and participates in a dynamic computation graph. Each operation creates a new node in the graph with references to its inputs, allowing gradients to be propagated backwards through the graph during a `.backward()` call.

**Computation graph:** Every array tracks its `dependencies` (the arrays it was computed from) and an `order` counter that determines the reverse topological sort used during backprop. The `backward()` method performs a BFS over all transitive dependencies, sorts them by reverse creation order, zeros all gradient accumulators, and calls each node's `grad_fn` in sequence.

**Implemented operations and their gradient functions:**
- **Elementwise:** `+`, `-`, `*`, `/`, `exp`, `log` — all with NumPy broadcasting support via an `_unbroadcast_to` helper that sums out broadcasted dimensions
- **Linear algebra:** `@` (matrix multiply), `sum`, `reshape`, `transpose` — supporting full gradient flow through shape-changing operations
- **Operator overloading:** `__add__`, `__mul__`, `__matmul__`, etc. so expressions like `a @ b + c` build the graph automatically

**Gradient verification:** Gradients are validated against central-difference numerical estimates (`numerical_grad`) at scalar, vector (shape `(5,)`), and high-dimensional (d=1000) inputs. For d=1000, backprop computes the full gradient in a single pass versus 2000 forward evaluations for the numerical method, demonstrating the asymptotic advantage of reverse-mode autodiff.

**Setup:**
```bash
pip install -r requirements.txt
python main.py
```

---

## Part 2 — Neural Network Training on MNIST

Trains and benchmarks deep neural networks on the MNIST digit classification dataset using PyTorch. Implements multiple architectures and optimizer configurations, with systematic hyperparameter search to understand their effects on convergence and generalization.

**Architectures:**
- **Fully connected MLP:** Two hidden layers (784 → 1024 → 256 → 10) with ReLU activations
- **MLP with Batch Normalization:** Same architecture with `BatchNorm1d` after each linear layer, improving training stability
- **CNN:** Four convolutional layers (16 and 32 channels, 3×3 kernels) with batch norm, ReLU, and max pooling, followed by two fully connected layers — capturing spatial structure in the 28×28 images

**Optimizers compared:**
- Vanilla SGD
- SGD with momentum (β=0.9)
- Adam (ρ₁=0.99, ρ₂=0.999)

All three are benchmarked on wall-clock training time and test accuracy over 10 epochs, with per-epoch loss and accuracy curves saved as plots.

**Hyperparameter search (Part 2):**
- **Grid search** over learning rate α ∈ {1.0, 0.3, ..., 0.001}, momentum β ∈ {0.8, 0.9, 0.95}, and weight decay ∈ {0, 1e-4, 1e-3} — 45 total configurations
- **Random search** over the same space using log-uniform sampling for α and uniform sampling for β across 15 trials
- Both methods split training data into 90/10 train/validation splits for model selection, then report final test accuracy

**Setup:**
```bash
pip install -r requirements.txt
python main.py
```

---

## Part 3 3 — LLM Inference Engine

A complete transformer inference engine for [OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct), a 1B-parameter open language model from AllenAI, written using only low-level PyTorch tensor operations — no `torch.nn` layers used in the forward pass.

**Memory management:** A custom `BumpAllocator` pre-allocates a large contiguous tensor and hands out views into it for temporary activations. A `BumpAllocatorScope` context manager marks a high-water position on entry and resets `bytes_allocated` on exit, allowing temporary buffers allocated inside a `with alloc.scope()` block to be freed automatically without any garbage collection overhead. This avoids repeatedly allocating and freeing tensors during inference.

**Transformer forward pass (`Olmo2Model`):**
- **Token embedding lookup** via `torch.index_select` into the embedding weight matrix, written into a preallocated buffer
- **KV cache:** Pre-allocated key and value caches of shape `(num_layers, num_heads, cache_seqlen, head_size)` to support autoregressive generation without recomputing past tokens
- **RMSNorm (`layer_norm_`):** Computed in fp32 to avoid precision loss, using the bump allocator for temporary buffers. Mutates the input tensor in-place
- **Rotary Positional Embeddings (`apply_rope_`):** RoPE matrices are precomputed for all positions and stored as `(max_position_embeddings, head_size//2, 2, 2)` rotation matrices. Applied via batched matrix multiplication over the query and key tensors
- **Causal self-attention (`self_attn_`):** Q/K/V projections, per-head Q/K norm, RoPE application, scaled dot-product attention with numerically stable softmax (subtract max before exp), context aggregation via `torch.bmm`, output projection, residual connection, and post-attention RMSNorm — all using preallocated buffers
- **MLP block (`mlp_`):** SiLU-gated feed-forward network (up proj × gate proj → SiLU → elementwise product → down proj) with residual and post-feedforward RMSNorm
- **Greedy decoding:** Autoregressive generation by repeatedly taking the argmax of the final token's logits and feeding it back as the next input token

**Setup:**
```bash
pip install -r requirements.txt
# Also works on Google Colab without any additional setup
python main.py
```
