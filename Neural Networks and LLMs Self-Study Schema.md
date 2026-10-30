# Neural Networks and LLMs: Self-Study Schema

Oct 20, 2026 · @Valerio Gianella

## How to read this schema

Six blocks, about 300 to 410 hours in total, almost all free; the 2,000 USD budget is a reserve, the schema spends under 300 USD of it. Block 0 to 3 are the 2-month core (about 200 h at 25 h/week); blocks 4 to 6 are the extension (about 100 to 180 h).

Criteria applied to every row, from the two reviews:

1. Oracle: the resource has tests, reference numbers or a differential target against a PyTorch primitive, so a solo learner can tell right from wrong.
2. Tacit over derivable: videos only for content you cannot derive in 15 minutes; derivable objects are derived on paper first, then implemented as confirmation.
3. One spine: your own repository, grown stage by stage; courses and books are pulled in, not followed end to end.
4. Current where the field moved (post-training, data, inference, evaluation), timeless elsewhere; no novelty before its baseline.
5. Artefact per unit: code in the repo, a note, or an experiment table; video is capped at 25 percent of hours.

Columns: Resource = what to use, in order; Link = where to find it; Date = when you engaged with or completed the row, left blank until then. Below each table, one line per subject: Comments = how to use it and what to skip; after the divider, Optional = material around the topic that is not part of the course.

Abbreviations: DD = Karpathy Deep Dive into LLMs; Z2H = Karpathy Neural Networks Zero to Hero; UDL = Prince, Understanding Deep Learning; CS336 = Stanford Language Modeling from Scratch; a1, a5 = its assignments; TRL = Hugging Face transformer RL library.

## Block 0: Primer and tooling (10 to 15 h)

Goal: a map of the whole LLM pipeline, PyTorch in your hands, and your own repository started.

| Subject | Resource | Link | Date |
| --- | --- | --- | --- |
| Map of the field | Karpathy, Deep Dive into LLMs like ChatGPT (3.5 h video) | [Karpathy YouTube channel](https://www.youtube.com/@AndrejKarpathy) | 21.10 |
| Statistics to deep learning map | Your own 2-page note | none | 22.10 |
| Matrix calculus | Parr and Howard, The Matrix Calculus You Need for Deep Learning | [explained.ai](https://explained.ai/matrix-calculus/) | 26.10 |
| PyTorch | PyTorch 60-minute blitz, then einops | [Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), [einops](https://einops.rocks/) |  |
| Your repository | A GitHub repo, pytest, uv or pip, pre-commit | [GitHub](https://github.com/), [uv](https://docs.astral.sh/uv/) | 21.10 |

**Comments**

- **Map of the field**: Watch once, in full, before anything else. Pretraining, SFT, RLHF, tokenisation, hallucination, all in one pass. Write a one-page map from memory afterwards.
- **Statistics to deep learning map**: Write it yourself: LM = autoregressive conditional density; cross-entropy = NLL; perplexity = exp(mean NLL); SGD = Robbins Monro; Adam = diagonally preconditioned SGD; weight decay = Gaussian prior (approximate under AdamW); temperature = tempering; DPO = Bradley Terry MLE; GRPO = REINFORCE with group baseline; scaling laws = power-law regression. 3 to 4 h.
- **Matrix calculus**: 30-minute skim for notation; you know the content.
- **PyTorch**: Tensors, autograd, nn.Module, a training loop. 3 to 4 h. Use einops for every reshape from day one.
- **Your repository**: The spine of the course. Every block adds a module and a test. Commit weekly at minimum. 1 h setup.

---

**Optional**

- **Map of the field**: Karpathy, Intro to LLMs (1 h, 2023) on the same channel: older, shorter version of the same map
- **Statistics to deep learning map**: Murphy, Probabilistic Machine Learning: An Introduction (free PDF) ch 1, 4, 8 if you want the mapping done for you
- **Matrix calculus**: Petersen and Pedersen, The Matrix Cookbook, as a lookup table
- **PyTorch**: PyTorch internals (ezyang blog) if you want to know how autograd is implemented
- **Your repository**: Weights and Biases free tier for run tracking, or plain CSV plus matplotlib

## Block 1: Foundations (40 to 50 h)

Goal: backprop, MLPs, optimisation and regularisation implemented, tested and debugged from loss curves. Derivable parts on paper first; tacit parts from Karpathy.

| Subject | Resource | Link | Date |
| --- | --- | --- | --- |
| Backprop and autograd | Self-derivation on paper, then a micrograd-style scalar autograd in your repo with a finite-difference gradient check | [micrograd repo](https://github.com/karpathy/micrograd) as the diff target after your attempt |  |
| MLP training dynamics | Z2H makemore part 3: activations, gradients, BatchNorm | [Z2H page](https://karpathy.ai/zero-to-hero.html) |  |
| Optimisation | Implement SGD with momentum, Adam and AdamW yourself; differential test against torch.optim on a toy quadratic | [UDL ch 6](https://udlbook.github.io/udlbook/) |  |
| Regularisation and generalisation | UDL ch 8 and 9 read fast; experiments on your MLP: weight decay, dropout, early stopping, label smoothing | [UDL PDF](https://udlbook.github.io/udlbook/) |  |
| Debugging from curves | Injected-bug exercise on your own MLP: wrong init scale, forgotten zero\_grad, learning rate 10x too high, data leakage | your repo |  |
| Reference text | UDL, Prince (free PDF) | [udlbook.github.io](https://udlbook.github.io/udlbook/) |  |

**Comments**

- **Backprop and autograd**: Chain rule on a DAG; derive in under an hour, implement in 3 to 4 h, check against finite differences and against torch.autograd. Skip the Z2H micrograd video, it is derivable.
- **MLP training dynamics**: The one video with content you cannot derive: what dead activations look like, why init scale matters, how to read gradient and update histograms. Build along, 6 to 8 h. Skip makemore parts 1, 2, 4, 5.
- **Optimisation**: Why Adam: diagonal preconditioning with EMA second moments. Why AdamW: decoupled decay. Cosine schedule with warmup. 4 to 5 h.
- **Regularisation and generalisation**: Read for argument, not vocabulary; your statistics covers bias-variance. Run the experiments, 4 to 5 h.
- **Debugging from curves**: Diagnose each from loss and gradient curves only, before looking at the code. 3 to 4 h. This is the skill no lecture teaches.
- **Reference text**: Consult ch 6, 7, 9, 11, 12 on demand; do not read linearly. Ch 3 to 5 are below your level.

---

**Optional**

- **Backprop and autograd**: UDL ch 7 (Gradients and initialization) as the reference; the Z2H micrograd video if you get stuck
- **MLP training dynamics**: makemore parts 1 and 2 as text only if you want the bigram and MLP setup; part 4 (manual backprop) is a good 2 h exercise, not required
- **Optimisation**: Loshchilov and Hutter, Decoupled Weight Decay (AdamW paper, arXiv 1711.05101)
- **Regularisation and generalisation**: UDL ch 20 (Why does deep learning work) for the open questions
- **Debugging from curves**: Karpathy, A Recipe for Training Neural Networks (blog, 2019)
- **Reference text**: d2l.ai (Dive into Deep Learning) as the code-first alternative reference; Fleuret, The Little Book of Deep Learning (free PDF, 160 pages) as a compact overview

## Block 2: Transformers and LLMs (150 to 200 h)

Goal: a modern transformer LM (RMSNorm, RoPE, SwiGLU, pre-norm) with BPE tokenizer, AdamW training loop, SFT, DPO or GRPO, KV-cache inference and an eval harness, all in your repo, passing the CS336 a1 tests and reaching its reference loss on TinyStories.

| Subject | Resource | Link | Date |
| --- | --- | --- | --- |
| First transformer build | Z2H, Let's build GPT from scratch (2 h video) | [Z2H page](https://karpathy.ai/zero-to-hero.html) |  |
| Tokenisation | Z2H, Let's build the GPT tokenizer (2 h); then CS336 a1 BPE under its tests | [Z2H page](https://karpathy.ai/zero-to-hero.html), [CS336 a1](https://github.com/stanford-cs336/assignment1-basics) |  |
| Modern transformer block under tests | CS336 assignment 1: RMSNorm, RoPE, SwiGLU, pre-norm transformer, cross-entropy, AdamW, schedule, training loop | [CS336 a1 repo](https://github.com/stanford-cs336/assignment1-basics), [CS336 site](https://cs336.stanford.edu/) |  |
| Pretraining run | CS336 a1 TinyStories run to the handout reference loss; one OpenWebText-subsample run if compute allows | [CS336 a1 handout](https://github.com/stanford-cs336/assignment1-basics) |  |
| Theory lectures | CS336 Spring 2026 lectures 1 to 9, plus scaling, data, inference, alignment lectures | [CS336 site](https://cs336.stanford.edu/), [Stanford Online YouTube](https://www.youtube.com/@stanfordonline) |  |
| Scaling laws | CS336 scaling lectures; Hoffmann et al., Chinchilla; your own 4-size fit on TinyStories | [Chinchilla, arXiv 2203.15556](https://arxiv.org/abs/2203.15556), [Kaplan, arXiv 2001.08361](https://arxiv.org/abs/2001.08361) |  |
| Data | CS336 data lecture; hands-on filter and dedupe on the a1 OpenWebText subsample; measure the effect on loss | [CS336 site](https://cs336.stanford.edu/), [nanochat dataset code](https://github.com/karpathy/nanochat) |  |
| SFT | nanochat chat\_sft.py for the mechanism (chat template, loss masking); Smol Course instruction-tuning module with TRL for the tooling | [nanochat](https://github.com/karpathy/nanochat), [Smol Course](https://github.com/huggingface/smol-course), [TRL docs](https://huggingface.co/docs/trl) |  |
| Preference optimisation and RL | DPO from scratch via Raschka's notebook; GRPO via CS336 a5 or Raschka's Reasoning from Scratch companion; DeepSeek-R1 paper after | [Raschka repo](https://github.com/rasbt/LLMs-from-scratch), [CS336 a5](https://github.com/stanford-cs336/assignment5-alignment), [DPO, arXiv 2305.18290](https://arxiv.org/abs/2305.18290), [GRPO in DeepSeekMath, arXiv 2402.03300](https://arxiv.org/abs/2402.03300), [R1, arXiv 2501.12948](https://arxiv.org/abs/2501.12948) |  |
| Inference and KV cache | Implement a KV cache in your own model (Raschka notebook as guide); then read nanochat engine.py; CS336 inference lecture for the arithmetic intensity | [Raschka KV-cache notebook](https://github.com/rasbt/LLMs-from-scratch), [nanochat](https://github.com/karpathy/nanochat) |  |
| Evaluation | CS336 eval lecture; run lm-evaluation-harness or nanochat base\_eval on your own model; build one custom eval with confidence intervals | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [nanochat](https://github.com/karpathy/nanochat) |  |
| Reading current models | Llama 3 paper sections 3 and 4; DeepSeek-V3 architecture section; Qwen3 technical report | [Llama 3, arXiv 2407.21783](https://arxiv.org/abs/2407.21783), [DeepSeek-V3, arXiv 2412.19437](https://arxiv.org/abs/2412.19437) |  |
| Capstone | Ablation study on your own model anchored to the a1 reference loss: remove RoPE, RMSNorm, SwiGLU, pre-norm; vary LR, warmup, batch; one data filter on or off; predictions written before each run | your repo |  |
| Guided book alternative | Raschka, Build a Large Language Model (From Scratch), Manning, about 40 USD; code free | [Repo](https://github.com/rasbt/LLMs-from-scratch), [Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch) |  |

**Comments**

- **First transformer build**: Build along, then rewrite from memory. Attention as softmax-weighted average, causal mask, residual stream, LayerNorm. 8 to 10 h with the rewrite.
- **Tokenisation**: The video explains, a1 checks. Differential test your BPE against tiktoken on identical merges. 8 to 10 h.
- **Modern transformer block under tests**: The hardening pass: your Z2H build rebuilt to pass the pytest suite. Differential tests: your attention vs F.scaled\_dot\_product\_attention, your RMSNorm vs torch.nn.RMSNorm. 30 to 50 h. The only large assignment with an oracle; do it in full at 25 h/week, tokenizer and model modules only at 10 h/week.
- **Pretraining run**: Reference loss is your oracle for the whole stack. TinyStories fits a T4 in under an hour. 6 to 10 h including debugging.
- **Theory lectures**: Watch with the IDE open, after the corresponding build, not before. About 25 h of video; take the ones tied to your current stage. Skip the kernel and parallelism lectures.
- **Scaling laws**: A power-law regression with selection effects; you will see through it. Fit 4 model sizes, predict the fifth, report the error. 6 to 8 h. Do not attempt CS336 a3 (needs a Stanford-only training API).
- **Data**: Every plan under-serves this. One experiment: train with and without a quality filter, same compute. 6 to 8 h.
- **SFT**: Implement loss masking in your own model first, then use TRL on a small open model (SmolLM3 or Qwen3 1.7B). 10 to 12 h.
- **Preference optimisation and RL**: DPO is one afternoon once you see it as Bradley Terry MLE. GRPO on a small verifiable task (arithmetic) is the real exercise. 15 to 25 h. a5 only at 25 h/week.
- **Inference and KV cache**: Reading alone does not teach the shape bookkeeping; implement it. Then 4-bit quantisation of a small model with bitsandbytes or llama.cpp. 8 to 10 h.
- **Evaluation**: Your statistics is the advantage here: CIs on accuracy with small n, contamination, prompt sensitivity. 6 to 8 h.
- **Reading current models**: Active protocol only: predict the ablation table before reading it, or implement one component (GQA or MLA) in your model with a before-and-after. 6 to 10 h.
- **Capstone**: 25 h at 10 h/week, 50 h at 25 h/week. One note per ablation: hypothesis, result, explanation. This is the integration test and the best learning per hour.
- **Guided book alternative**: Only at 10 h/week if you prefer a guided sequential path over Z2H plus a1; otherwise use its notebooks (DPO, KV cache, Qwen3, MoE) as targeted modules.

---

**Optional**

- **First transformer build**: Vaswani et al., Attention Is All You Need (arXiv 1706.03762); Alammar, The Illustrated Transformer (blog)
- **Tokenisation**: minbpe repo (Karpathy); tiktoken source
- **Modern transformer block under tests**: CS336 a2 (Triton kernels, distributed): systems depth, not needed for the goal
- **Pretraining run**: Z2H, Let's reproduce GPT-2 (124M), 4 h video on the Karpathy channel: mixed precision, torch.compile, flash attention, gradient accumulation; watch selectively
- **Theory lectures**: CS224N 2024 public lectures on attention, pretraining and RLHF (older, 2024 videos) at [web.stanford.edu/class/cs224n](https://web.stanford.edu/class/cs224n/)
- **Scaling laws**: muP paper (Yang et al.) in the depth phase
- **Data**: CS336 a4 (data) in the depth phase; FineWeb and FineWeb-Edu dataset cards on Hugging Face
- **SFT**: Hugging Face LLM Course (tokenizers, Trainer, fine-tuning) at [huggingface.co/learn/llm-course](https://huggingface.co/learn/llm-course)
- **Preference optimisation and RL**: Smol Course preference-alignment and RL modules as lighter versions; Lilian Weng, RLHF and reward hacking posts
- **Inference and KV cache**: vLLM docs on paged attention and continuous batching
- **Evaluation**: CS224N Winter 2026 a4 (LLM benchmarking) in the depth phase; Smol Course evaluation module
- **Reading current models**: Lilian Weng, The Transformer Family v2 (blog); Raschka's Llama 3.2 and Qwen3 from-scratch notebooks
- **Capstone**: Replicate one paper figure (a DPO beta sweep, a scaling fit) as a second capstone
- **Guided book alternative**: Raschka, Build a Reasoning Model (From Scratch), the GRPO companion, in the depth phase

## Block 3: Engineering and empirics (30 to 40 h, woven into blocks 1 and 2)

Goal: the practice that separates a working run from a notebook demo. Not a separate phase: each row is done inside the block it belongs to.

| Subject | Resource | Link | Date |
| --- | --- | --- | --- |
| Mixed precision, compile, flash attention | Z2H, Let's reproduce GPT-2 (124M), the engineering sections | [Karpathy YouTube channel](https://www.youtube.com/@AndrejKarpathy) |  |
| Experiment discipline | Your own run tracker: config, seed, git hash, loss curve, eval, per run | CSV plus matplotlib, or [Weights and Biases](https://wandb.ai/) free tier |  |
| Reading training reports | Llama 3, DeepSeek-V3 and R1 training sections | links in block 2 |  |
| Interpretability basics | Anthropic, A Mathematical Framework for Transformer Circuits; one ARENA exercise notebook on induction heads | [Transformer Circuits](https://transformer-circuits.pub/), [ARENA](https://arena.education/) |  |
| Compute arithmetic | CS336 lecture on FLOPs and memory; your own calculation for your model | [CS336 site](https://cs336.stanford.edu/) |  |

**Comments**

- **Mixed precision, compile, flash attention**: bf16 autocast, torch.compile, F.scaled\_dot\_product\_attention, gradient accumulation, measured step time. Apply each to your a1 model and record the speedup. 6 to 8 h.
- **Experiment discipline**: Pre-register the hypothesis before each run. Keep a claims log: one test or experiment per claim you believe. 2 to 3 h setup.
- **Reading training reports**: What they spend pages on (data, infrastructure, failures) is what matters in practice. 4 to 6 h, active protocol.
- **Interpretability basics**: 6 to 8 h, enough to know what a residual stream, an attention head and an induction head are mechanistically.
- **Compute arithmetic**: 6ND FLOPs, parameters times bytes for memory, KV cache size per token. Do it for your model and for Llama 3 8B. 2 to 3 h.

---

**Optional**

- **Mixed precision, compile, flash attention**: PyTorch AMP docs; Dao et al., FlashAttention (arXiv 2205.14135)
- **Experiment discipline**: Hydra or a plain dataclass config
- **Reading training reports**: OLMo 2 report (Allen AI) for a fully open training recipe
- **Interpretability basics**: Neel Nanda, TransformerLens tutorials; Anthropic scaling monosemanticity paper
- **Compute arithmetic**: Transformer Inference Arithmetic (kipply blog)

## Block 4: Generative models, VAEs, diffusion, flow matching (45 to 65 h)

Goal: derive the ELBO, the denoising objective and the flow-matching objective; train a small diffusion or flow model on MNIST or CIFAR-10 in your repo; reuse your transformer as the backbone (a DiT). The most probabilistic family, so your statistics does most of the work.

| Subject | Resource | Link | Date |
| --- | --- | --- | --- |
| Convolutional networks | UDL ch 10; derive conv as a weight-tied Linear layer; implement a minimal conv U-Net on MNIST | [UDL PDF](https://udlbook.github.io/udlbook/) |  |
| VAEs and the ELBO | UDL ch 17; derive the ELBO yourself; implement a VAE on MNIST | [UDL PDF](https://udlbook.github.io/udlbook/) |  |
| Diffusion theory | Lilian Weng, What are Diffusion Models (blog); UDL ch 18; DDPM paper | [Weng post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), [DDPM, arXiv 2006.11239](https://arxiv.org/abs/2006.11239) |  |
| Diffusion practice | Hugging Face Diffusion Models course, units 1 and 2; implement DDPM from scratch on MNIST, then with your transformer as backbone | [HF diffusion course](https://github.com/huggingface/diffusion-models-class) |  |
| Flow matching | MIT 6.S184 Introduction to Flow Matching and Diffusion Models (lecture notes and labs); Lipman et al. paper | [MIT 6.S184](https://diffusion.csail.mit.edu/), [Flow Matching, arXiv 2210.02747](https://arxiv.org/abs/2210.02747) |  |
| Where it is used | Read one applied paper from a field you care about: GenCast (weather), AlphaFold 3 structure module, or Stable Diffusion 3 | [GenCast, Nature 2024](https://www.nature.com/articles/s41586-024-08252-9) |  |

**Comments**

- **Convolutional networks**: Prerequisite for diffusion practice: the HF course and most DDPM implementations use a conv U-Net, and the DiT patchify step is a strided conv. Derive on paper: output size (n − k + 2p)/s + 1, parameter count, and conv as a Linear layer on unfolded patches with shared weights; oracle: your F.unfold plus matmul against nn.Conv2d with torch.allclose. Then build a small U-Net (two down, two up, skip connections) to reuse as the DDPM backbone. 4 to 6 h.
- **VAEs and the ELBO**: Variational inference you already know, with an encoder network. 6 to 8 h. Keep the latent space 2-D so you can plot it.
- **Diffusion theory**: Forward process as a Gaussian Markov chain, reverse as learned denoising, loss as a weighted score-matching objective. Derive the simplified loss yourself. 8 to 10 h.
- **Diffusion practice**: From-scratch first, diffusers library second. Sampling steps, noise schedule, classifier-free guidance as experiments. 12 to 15 h.
- **Flow matching**: The 2024+ formulation used by Flux, SD3 and most new image and video models: regress a velocity field, sample with an ODE solver. 10 to 12 h with the labs.
- **Where it is used**: 3 to 4 h, active reading: identify the backbone, the objective, the conditioning.

---

**Optional**

- **Convolutional networks**: Stanford CS231n notes on ConvNets (Karpathy's original course); Ronneberger et al., U-Net (arXiv 1505.04597); LeCun et al., Gradient-Based Learning Applied to Document Recognition (1998) for LeNet-5
- **VAEs and the ELBO**: Kingma and Welling, Auto-Encoding Variational Bayes (arXiv 1312.6114)
- **Diffusion theory**: Song et al., Score-Based Generative Modeling through SDEs (arXiv 2011.13456) for the continuous-time view
- **Diffusion practice**: Peebles and Xie, Scalable Diffusion Models with Transformers (DiT, arXiv 2212.09748)
- **Flow matching**: Meta's Flow Matching Guide and Code (arXiv 2412.06264) as the comprehensive reference
- **Where it is used**: Audio: Stable Audio or MusicGen reports

## Block 5: Sequence models, RNNs, LSTMs, state-space models (20 to 30 h)

Goal: one RNN and one LSTM built and trained on the same data as your transformer; understand why attention won and where SSMs come back. Short block, mostly comparison.

| Subject | Resource | Link | Date |
| --- | --- | --- | --- |
| RNNs and LSTMs | Olah, Understanding LSTM Networks (blog); UDL ch 12.1 to 12.3 background; implement an LSTM language model in your repo on TinyStories | [Olah post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), [UDL PDF](https://udlbook.github.io/udlbook/) |  |
| State-space models | Mamba paper; The Annotated S4 (blog with code); run a small Mamba via the mamba-ssm or a pure-PyTorch reimplementation | [Mamba, arXiv 2312.00752](https://arxiv.org/abs/2312.00752), [Annotated S4](https://srush.github.io/annotated-s4/) |  |
| Why transformers won, and the current debate | CS336 architecture lecture; one hybrid-model report (Jamba, or a Nvidia Nemotron-H report) | [CS336 site](https://cs336.stanford.edu/) |  |

**Comments**

- **RNNs and LSTMs**: Same data, same tokenizer, same eval as your transformer. Compare loss per FLOP and training stability; observe vanishing gradients directly. 8 to 10 h.
- **State-space models**: Selective state space as a linear-time attention substitute; the parallel scan is the engineering trick. 8 to 10 h.
- **Why transformers won, and the current debate**: Write a one-page note: inductive bias, parallelism, in-context learning, memory per token. 3 to 4 h.

---

**Optional**

- **RNNs and LSTMs**: d2l.ai ch 9 and 10 for a code-first treatment; Karpathy, The Unreasonable Effectiveness of RNNs (2015 blog)
- **State-space models**: Mamba 2 paper (state space duality); Jamba report for a hybrid architecture
- **Why transformers won, and the current debate**: Time-series foundation models (Chronos, TimesFM) if forecasting interests you

## Block 6: Tabular and structured data (20 to 30 h)

Goal: know honestly when a neural network beats a GLM or gradient boosting on tabular data, and how to get calibrated distributional outputs from one. Mostly experiments on data you understand.

| Subject | Resource | Link | Date |
| --- | --- | --- | --- |
| Nets vs trees, the evidence | Grinsztajn et al., Why do tree-based models still outperform deep learning on tabular data; Gorishniy et al., Revisiting Deep Learning Models for Tabular Data | [Grinsztajn, arXiv 2207.08815](https://arxiv.org/abs/2207.08815), [Gorishniy, arXiv 2106.11959](https://arxiv.org/abs/2106.11959) |  |
| Categorical embeddings and architecture | Implement an MLP with learned embeddings for categoricals, numeric normalisation, residual blocks; compare to one-hot | your repo |  |
| Distributional outputs and calibration | Output a Poisson, gamma or mixture distribution via a parametric head; proper scoring rules; reliability diagrams; Guo et al. on calibration | [Guo, arXiv 1706.04599](https://arxiv.org/abs/1706.04599) |  |
| Insurance-flavoured benchmark | Wuthrich and Merz, Statistical Foundations of Actuarial Learning (open access), ch 7 on feed-forward networks; Holvoet, Antonio and Henckaerts benchmark on the freMTPL2 data | [Wuthrich and Merz, Springer](https://link.springer.com/book/10.1007/978-3-031-12409-9), [Holvoet et al., arXiv 2310.12671](https://arxiv.org/abs/2310.12671) |  |

**Comments**

- **Nets vs trees, the evidence**: Read both, then reproduce one comparison: MLP with embeddings vs LightGBM or XGBoost on one open dataset. 6 to 8 h.
- **Categorical embeddings and architecture**: The one design choice that matters on tabular data. 4 to 5 h.
- **Distributional outputs and calibration**: Your home ground: NLL as the loss, deviance as the metric, temperature scaling as post-hoc calibration. 6 to 8 h.
- **Insurance-flavoured benchmark**: Optional inside the block, but the best checkable experiment for you: GLM vs FFNN vs CANN with a published result to compare against. 8 to 10 h.

---

**Optional**

- **Nets vs trees, the evidence**: TabPFN v2 (Nature 2025) for the foundation-model approach to small tables
- **Categorical embeddings and architecture**: FT-Transformer from Gorishniy as the attention-based variant
- **Distributional outputs and calibration**: Conformal prediction (Angelopoulos and Bates, gentle introduction, arXiv 2107.07511) for distribution-free intervals
- **Insurance-flavoured benchmark**: Kuo, DeepTriangle (arXiv 1804.09253) for reserving; Wuthrich's neural chain-ladder papers

## Budget, compute and tools

Planned spend about 200 to 300 USD of the 2,000 available. Everything with learning value is free; money buys compute and, optionally, two books.

| Item | Cost (USD) | Needed for |
| --- | --- | --- |
| Colab Pro, 2 to 3 months | 10 per month | Blocks 1 to 2 small runs, TinyStories, SFT on a 1.7B model |
| Rented GPU hours (RunPod, Lambda, Vast) | 1.5 to 3.5 per H100 hour, about 100 to 150 total | CS336 a1 OpenWebText run (about 1.5 H100 hours), a5 GRPO (a few A100 hours), diffusion training, ablation sweeps |
| Raschka, Build a Large Language Model (From Scratch) | about 40 | Block 2, only if you choose the guided path |
| Frontier model API credits | 20 to 50 | Adversarial code review of each implementation, synthetic data for eval experiments |
| Reserve | about 1,500 unspent | A paid cohort with office hours if one is running when a block needs feedback (an evaluation or LLM-engineering cohort on Maven, 1,000 to 2,000), or a second GPU budget for the depth phase |

**Comments**

- **Colab Pro, 2 to 3 months**: About 100 compute units per month, roughly 8 A100 hours; disconnects on long runs, so rent for anything over 2 h
- **Rented GPU hours (RunPod, Lambda, Vast)**: Book the hours as hard dates; they double as deadlines
- **Raschka, Build a Large Language Model (From Scratch)**: Code is free on GitHub either way
- **Frontier model API credits**: Prompt it to find bugs and write a failing test, never to confirm correctness
- **Reserve**: Spend only if it buys feedback; content is never the bottleneck

Tools: Python 3.15, PyTorch 2.x, einops, pytest, tiktoken for differential tests, Hugging Face transformers, datasets, TRL, PEFT for LoRA, bitsandbytes or llama.cpp for quantisation, lm-evaluation-harness, matplotlib. Avoid frameworks that hide the training loop (Lightning, Keras) until block 2 is done.

## Self-tests and schedule

Two-month core = blocks 0 to 3 at 25 h/week (about 200 h); the same at 10 h/week takes about 4.5 months with CS336 a1 reduced to the tokenizer and model modules and a5 dropped. Extension = blocks 4 to 6, about 100 to 180 h, in the order 4, 5, 6 or by interest.

| Week (25 h/week) | Block | Milestone | Date |
| --- | --- | --- | --- |
| 1 | 0, 1 | Map written, stats-to-DL note written, scalar autograd passing finite-difference checks | |
| 2 | 1, 2 | makemore 3 built along, Let's build GPT rebuilt from memory, tokenizer video done | |
| 3 to 5 | 2, 3 | CS336 a1 tests green, TinyStories reference loss reached, CS336 lectures 1 to 9 watched with IDE open | |
| 6 | 2, 3 | GPT-2 engineering applied with measured speedups, 4-size scaling fit, one data-filter experiment | |
| 7 | 2 | SFT with loss masking in own model, DPO from scratch, GRPO on a verifiable task | |
| 8 | 2, 3 | KV cache implemented, eval harness run, ablation capstone written up | |
| 9 to 11 | 4 | VAE, DDPM from scratch, flow matching labs, DiT backbone | |
| 12 | 5 | LSTM vs transformer comparison, Mamba run, one-page note | |
| 13 to 14 | 6 | Tabular benchmark vs GBM, distributional head with calibration, optional insurance benchmark | |

Self-test at 2 months, all checkable alone:

- [ ] All CS336 a1 tests pass; TinyStories loss within the handout tolerance
- [ ] Attention, RMSNorm, RoPE and AdamW written from memory, each with a differential test against the PyTorch primitive
- [ ] Three injected bugs diagnosed from loss curves alone
- [ ] One page: MLE, KL and Bradley Terry derivations behind cross-entropy, RLHF and DPO
- [ ] Ablation table with predictions written before the runs, one paragraph per surprise
- [ ] KV-cache memory and compute trade-off computed for your own model and for Llama 3 8B

Self-test at 4 to 6 months:

- [ ] DDPM and a flow-matching model trained on MNIST with your transformer as backbone; samples and loss curves in the repo
- [ ] LSTM and transformer compared on the same data with loss per FLOP
- [ ] Tabular MLP vs GBM comparison reproduced from a published paper, with a calibrated distributional head
- [ ] One component from a current model (GQA or MLA) implemented with a before-and-after result
- [ ] A scaling law fitted on your own runs and its assumptions critiqued in one page
- [ ] One evaluation run with one documented way the benchmark could mislead

Rules that keep a solo learner honest: no video without the IDE open; video under 25 percent of hours; a claims log with one test per claim; weekly commit; one short note per block; GPU hours booked as dates.

Sources: the links in each table, plus the [CS336 course site](https://cs336.stanford.edu/), the [CS336 GitHub organisation](https://github.com/stanford-cs336), [Karpathy Zero to Hero](https://karpathy.ai/zero-to-hero.html), [Understanding Deep Learning](https://udlbook.github.io/udlbook/), [nanochat](https://github.com/karpathy/nanochat), [Hugging Face Smol Course](https://github.com/huggingface/smol-course), [Raschka LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch). Hour estimates are the reviewers' corrected figures, about 1.5 to 2.5 times video length; prices are as of October 2026 and should be checked before booking.
