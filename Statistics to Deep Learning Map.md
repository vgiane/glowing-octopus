# Statistics to Deep Learning Map

Block 0, row 2. Written from memory, no lookup. Two pages, 3 to 4 h.

For each row: the statistics object, the deep-learning name, the equation that makes them the same, and where the identity breaks.

## Abbreviations and named terms

Deep-learning side (left of each equals sign):

- LM: language model, a neural network that assigns a probability to the next token given the previous ones.
- Cross-entropy: the training loss used for classification and language models.
- Perplexity: the standard number reported to evaluate a language model.
- SGD: stochastic gradient descent, the basic optimizer, updates weights using the gradient of a random minibatch.
- Adam: adaptive moment estimation, the optimizer used in practice for almost all LLM training.
- AdamW: Adam with decoupled weight decay, the variant of Adam that applies weight decay directly to the weights instead of through the loss.
- Weight decay: shrinking the weights a little at every step, a form of regularization.
- Temperature: the number you divide the logits by before sampling a token; low is deterministic, high is random.
- DPO: direct preference optimization, a method to align a model to human preference pairs without training a separate reward model.
- GRPO: group relative policy optimization, a reinforcement-learning method for LLMs that compares several completions of the same prompt against each other.
- Scaling laws: empirical formulas that predict loss as a function of model size, data size and compute.

Statistics side (right of each equals sign):

- Autoregressive conditional density: a joint distribution factorized by the chain rule, each factor conditioned on everything before it.
- NLL: negative log-likelihood, minus the log probability of the data under the model; minimizing it is maximum likelihood estimation.
- Robbins Monro: the 1951 stochastic approximation algorithm, finds the root of an expectation using noisy samples with decaying step sizes.
- Preconditioned SGD: gradient descent where the gradient is multiplied by a matrix (the preconditioner) that rescales each direction; diagonal means the matrix only rescales each coordinate independently.
- Gaussian prior: putting a normal distribution centered at zero on the parameters; combined with the likelihood this gives MAP estimation.
- MAP: maximum a posteriori, the parameter value that maximizes prior times likelihood.
- Tempering: raising a distribution to the power one over T and renormalizing, from statistical physics and Bayesian computation.
- Bradley Terry: a model for pairwise comparisons where the probability that A beats B is the sigmoid of the difference of their scores.
- MLE: maximum likelihood estimation.
- REINFORCE: the score-function policy-gradient estimator from reinforcement learning, reward times the gradient of the log probability of the action.
- Baseline: a quantity subtracted from the reward in REINFORCE to reduce variance without changing the expected gradient.
- Power-law regression: fitting y = a times x to the power b, which is a straight line in log-log coordinates.

Prerequisites used in the derivations:

- Logits: the raw real-valued outputs of the network, one per vocabulary entry, before any normalization. Written z.
- Softmax: turns logits into a probability vector, p(v) = exp(z_v) / Σ_u exp(z_u). Exponentiate, then divide by the sum so the entries are positive and sum to one.
- Partition function Z: the normalizing constant Σ_u exp(z_u) in the softmax, or more generally the sum or integral that makes an unnormalized distribution sum to one.
- One-hot target: the observed token written as a probability vector with 1 at that token and 0 elsewhere; the Kronecker delta 1[v = x_t] seen as a distribution. All mass on one outcome.
- Entropy H(q): −Σ_v q(v) log q(v), the expected surprise under q; the lower bound of cross-entropy against q.
- KL: Kullback Leibler divergence, a measure of how different two distributions are.
- Unbiased estimator: a random quantity whose expectation equals the target. The minibatch gradient is an unbiased estimator of the full gradient.
- Hessian: the matrix of second derivatives of the loss. Describes local curvature; Newton's method divides the gradient by it.
- Fisher information F: E[∇log p ∇log pᵀ], the expected outer product of the score. Equals the expected Hessian of the NLL under the model, so it is a curvature matrix defined from the likelihood alone. Natural gradient divides the gradient by it.
- Prior, likelihood, posterior: p(θ) is what you believe about the parameters before data, p(D | θ) is the likelihood, and the posterior p(θ | D) ∝ p(D | θ) p(θ) is the update. MAP maximizes the posterior; MLE maximizes only the likelihood.
- RL: reinforcement learning.
- RLHF: reinforcement learning from human feedback, the general recipe DPO and GRPO belong to.
- PPO: proximal policy optimization, the RL algorithm GRPO is derived from.
- Policy π_θ(y | x): the LLM viewed as a distribution over completions y given prompt x. The reference policy π_ref is the frozen starting model. The reward r(x, y) is a scalar score for a completion.
- Score function ∇_θ log π_θ(y): the gradient of the log probability of a sample. Its expectation under π_θ is zero, which is what makes baselines free.
- Control variate: a quantity with known expectation subtracted from a Monte Carlo estimate to reduce variance without changing the mean. The baseline in REINFORCE is one.
- Least squares: fit parameters by minimizing the sum of squared residuals between data and model. Ordinary when the model is linear in the parameters, nonlinear otherwise.

---

## 1. LM = autoregressive conditional density

📊 **Statistics:** chain rule of probability, true for any joint distribution. Autoregressive means each factor is a parametric model of one step given the past, like an AR(p) process.

```
p(x_1..x_T) = Π_t p(x_t | x_1..x_{t-1})
```

🧠 **Deep learning:** a network maps a token prefix to logits over the vocabulary; softmax turns them into a categorical over the next token. Generation samples one token, appends it, repeats.

```
p_θ(x_t | x_<t) = softmax(f_θ(x_<t))[x_t]
```

🔗 **Why equal:** multiply the per-position conditionals and you get the chain-rule product. The network is only a parameterization of the factors, shared across positions. Token-by-token sampling is ancestral sampling from the joint. So an LM is a density estimator for sequences.

⚠️ **Where it breaks:** the identity is about the form of the model, not its fit. Two places where the form is narrower than the chain rule.

- Finite context window. The network conditions on the last k tokens, not on the whole past, so it is a k-th order Markov model. The product of its conditionals is still a valid joint, but it is a smaller family: no dependency longer than k can be represented. A fact stated at the start of a long document is invisible when predicting the end.
- Fixed left-to-right order. The chain rule holds for any ordering of the variables, but the network is trained on one. The joint is the same in principle, but the learned conditionals are not: a model trained on "A is B" does not automatically assign high probability to "B is A", because that requires a different factorization. It also cannot compute p(x_t | past and future) for infilling without a separate model.

---

## 2. Cross-entropy = NLL

📊 **Statistics:** negative log-likelihood of the data under the model. For a sequence, use the factorization of item 1, so it is a sum over positions. Minimizing it over θ is maximum likelihood.

```
NLL(θ) = −log p_θ(x_1..x_T) = −Σ_t log p_θ(x_t | x_<t)
```

🧠 **Deep learning:** the training loss, cross-entropy between a target distribution q and the softmax output p: −Σ_v q(v) log p(v). The target is one-hot on the token that actually came next.

🔗 **Why equal:** with a one-hot target every term vanishes except the true token, leaving minus the log probability the model gave it. That is the per-position NLL. The mean over tokens and minibatch is the number on every training curve, so minimizing cross-entropy is MLE of the autoregressive model.

```
CE(one_hot(x_t), p_θ) = −Σ_v 1[v = x_t] log p_θ(v | x_<t) = −log p_θ(x_t | x_<t)
```

In expectation over the true data distribution q:

```
E_q[−log p_θ] = H(q) + KL(q ‖ p_θ)
```

So minimizing cross-entropy is minimizing KL to the data, and the data entropy H(q) is a floor the loss can never go below. That floor is the irreducible constant in item 10.

⚠️ **Where it breaks:** exact for one-hot targets, because only then does the sum over the vocabulary collapse to one term.

- Label smoothing. The target becomes a mixture of the one-hot and the uniform, so extra terms survive in the sum. The loss is still a cross-entropy but no longer the log probability of any observed data, so it is not MLE; it is MLE plus a penalty that pulls probabilities toward uniform.
- Distillation. The target is a teacher's soft distribution. Minimizing cross-entropy against it minimizes KL to the teacher, not to the data. Likelihood of the data is not what is being maximized.
- Non-i.i.d. observations. Tokens in a sequence are dependent and the model conditions on them, so the loss is a likelihood but not an i.i.d. one. Classical MLE results that assume independent samples (standard errors, asymptotic normality) do not apply as stated, even though the point estimate is still the MLE.

---

## 3. Perplexity = exp(mean NLL)

📊 **Statistics:** the perplexity of a distribution is exp of its entropy, the number of equally likely outcomes that would have the same entropy. For a model scored on data it is exp of the average per-observation NLL, which is the geometric mean of 1/p over the observations.

```
PPL = exp( (1/T) Σ_t −log p_θ(x_t | x_<t) ) = ( Π_t 1/p_θ(x_t | x_<t) )^{1/T}
```

🧠 **Deep learning:** the evaluation number for a language model. Compute the mean cross-entropy per token on held-out text, exponentiate. In bits it is 2 to the mean cross-entropy.

🔗 **Why equal:** by definition, it is the mean loss of item 2 put through exp. The exp gives it a scale: a model that spreads probability uniformly over k candidates at every step has perplexity k, so PPL reads as the effective branching factor, the number of tokens the model is "choosing between" on average.

⚠️ **Where it breaks:** the number is well defined but easy to misuse.

- Tokenizer dependence. The log-likelihood of a whole text is invariant to tokenization (if it is lossless), but the mean per token is not: a tokenizer that splits text into more pieces has more, easier tokens. Perplexities from different tokenizers are not comparable. Compare per-character or per-byte NLL instead.
- Nonlinear scale. exp is convex, so differences in PPL do not translate linearly into differences in loss, and the average of per-document perplexities is not the perplexity of the corpus. Average the NLL first, then exponentiate.
- Not a measure of ability. It scores the model as a density estimator for the test text. A model fine-tuned to follow instructions gets worse perplexity on web text while being more useful. Downstream task metrics are a different quantity.

---

## 4. SGD = Robbins Monro

📊 **Statistics:** stochastic approximation (Robbins and Monro, 1951). You want the root θ* of a function g(θ) = E_ξ[G(θ, ξ)] but can only observe noisy evaluations G(θ, ξ). Iterate with decaying steps a_t. The iterates converge to θ* almost surely if the steps are large enough in total to reach anywhere but small enough in square to average out the noise, plus regularity conditions on g.

```
θ_{t+1} = θ_t − a_t G(θ_t, ξ_t),    Σ_t a_t = ∞,   Σ_t a_t² < ∞
```

🧠 **Deep learning:** the loss is an expectation over the data, L(θ) = E_x[ℓ(θ; x)]. A minibatch of B examples gives an unbiased estimate of its gradient. SGD moves against that estimate with learning rate η_t.

```
ĝ = (1/B) Σ_i ∇ℓ(θ; x_i),   E[ĝ] = ∇L(θ),   θ ← θ − η_t ĝ
```

🔗 **Why equal:** take G to be the minibatch gradient and g to be the full gradient ∇L. The root of ∇L is a stationary point of the loss, so SGD is Robbins Monro applied to the gradient. The learning-rate schedule is the sequence a_t, and the two conditions explain why schedules decay: a constant rate violates the second condition and converges only to a noise ball of radius proportional to η around the minimum, which is why training ends with the rate decayed toward zero.

⚠️ **Where it breaks:** the theorem is narrower than the practice.

- Nonconvexity. Robbins Monro assumes a unique root that g crosses monotonically, which for optimization means a convex loss. Neural losses have many stationary points, so the guarantee degrades to "converges to some stationary point", not to the minimizer.
- Finite horizon schedules. Warmup, cosine and linear decay to zero are chosen for a fixed budget of steps, not as infinite sequences satisfying the two sums. They work empirically; the theorem does not cover them.
- Modifications. Momentum, Adam's preconditioning, gradient clipping and weight decay all change the update beyond G, so the basic theorem does not apply as stated. Each has its own convergence analysis.

---

## 5. Adam = diagonally preconditioned SGD

📊 **Statistics:** a preconditioned gradient step multiplies the gradient by the inverse of a positive definite matrix P that rescales directions. Newton's method uses the Hessian. Natural gradient uses the Fisher information F = E[∇log p ∇log pᵀ]. A diagonal preconditioner keeps only the diagonal of P, so each coordinate gets its own step size η/d_i and the cross-terms are dropped.

```
θ ← θ − η P⁻¹ g,   diagonal case: θ_i ← θ_i − (η / d_i) g_i
```

🧠 **Deep learning:** Adam keeps two running averages per parameter, the gradient m (momentum) and the squared gradient v, bias-corrects them, and divides one by the square root of the other.

```
m ← β₁ m + (1−β₁) g,   v ← β₂ v + (1−β₂) g²,   θ ← θ − η m̂ / (√v̂ + ε)
```

🔗 **Why equal:** v is an estimate of E[g_i²] per coordinate, the diagonal of the uncentred second moment of the gradient, which is the empirical Fisher. So √v̂ is a diagonal preconditioner d_i and Adam is SGD with momentum under the preconditioner diag(√F_emp). The practical effect is that each coordinate moves by roughly η regardless of its gradient scale: coordinates with tiny gradients move as far as coordinates with large ones.

⚠️ **Where it breaks:** the preconditioner is a heuristic, not the Fisher.

- The square root. Natural gradient divides by F; Adam divides by √F. There is no derivation of the square root from curvature. It sits between no preconditioning and full natural gradient and is justified by working, which also means Adam is not a second-order method.
- Empirical Fisher. The true Fisher uses gradients with labels sampled from the model; the empirical version uses the observed labels. The two differ, and the empirical Fisher is known to be a poor curvature estimate. v also contains the gradient noise, so it preconditions by noise as much as by curvature.
- Diagonal only. Correlations between parameters are discarded. For a transformer, that ignores all the structure within a weight matrix.
- No fixed metric. v changes every step, so the update is not the gradient of any objective under a fixed metric. Adam can fail to converge on simple convex problems where SGD does; the fix (AMSGrad) makes the preconditioner monotone.

---

## 6. Weight decay = Gaussian prior (approximate under AdamW)

📊 **Statistics:** MAP estimation maximizes log likelihood plus log prior. An isotropic Gaussian prior on the parameters with variance σ² contributes minus the squared norm over 2σ². MAP with that prior is therefore NLL plus an L2 penalty, which is ridge regression or Tikhonov regularization with λ = 1/σ².

```
θ_MAP = argmin_θ  NLL(θ) + (λ/2) ‖θ‖²,   λ = 1/σ²
```

🧠 **Deep learning:** weight decay shrinks every weight toward zero by a factor (1 − ηλ) at each step, in addition to the gradient step.

```
θ ← θ − η (g + λθ) = (1 − ηλ) θ − η g
```

🔗 **Why equal:** the gradient of (λ/2)‖θ‖² is λθ. Under plain SGD, adding the L2 penalty to the loss produces exactly the weight-decay update. So SGD with weight decay λ is SGD on the MAP objective with a Gaussian prior of variance 1/λ, and the trained weights are a MAP estimate.

⚠️ **Where it breaks:** the exactness depends on the optimizer being plain SGD.

- Adam with L2 in the loss. The penalty gradient λθ gets divided by √v like every other gradient component, so coordinates with large gradient history are shrunk less. The effective prior is no longer isotropic and depends on the optimization trajectory, which is not a prior at all.
- AdamW. Loshchilov and Hutter fixed that by applying the shrink (1 − ηλ)θ outside the preconditioner. Now the loss gradient moves under the Adam metric and the decay under the identity metric, and the combined step is not the gradient of any single objective. There is no σ² such that AdamW computes a MAP estimate; the Gaussian prior is only an analogy and λ and η interact with the schedule in ways the Bayesian reading does not predict.
- Normalization layers. A weight matrix followed by LayerNorm or RMSNorm is scale invariant: multiplying it by a constant leaves the function unchanged. Decaying it does not regularize the function at all; it only shrinks the norm, which raises the effective learning rate. A prior on the weights is then not a prior on anything the model computes.
- Weights are not functions. Even under SGD, a Gaussian prior on weights of a deep nonlinear network induces a complicated prior on functions. The MAP identity is exact, but the "simplicity" it encodes is not the one you would pick on purpose.

---

## 7. Temperature = tempering

📊 **Statistics:** tempering raises a distribution to a power 1/T and renormalizes. In statistical physics it is the Boltzmann distribution at temperature T over energies E. In Bayesian computation it is the tempered posterior used in simulated annealing and parallel tempering. T → 0 concentrates all mass on the mode, T → ∞ flattens to uniform, and entropy increases monotonically in T.

```
p_T(x) ∝ p(x)^{1/T} = exp(−E(x)/T),   with E(x) = −log p(x)
```

🧠 **Deep learning:** sampling with temperature divides the logits z by T before the softmax. T = 1 is the model, T < 1 is "more deterministic", greedy decoding is the limit T → 0.

```
p_T(v) = softmax(z / T)_v = exp(z_v/T) / Σ_u exp(z_u/T)
```

🔗 **Why equal:** the softmax output is p(v) = exp(z_v) / Z, so z_v = log p(v) + log Z. Then exp(z_v/T) = p(v)^{1/T} · Z^{1/T}, and the constant cancels in the normalization. Dividing logits by T is exactly tempering the model's categorical, with the logits playing the role of negative energies. The physics name is literal, not an analogy.

⚠️ **Where it breaks:** the tempering is per token, not of the sequence distribution.

- Not the tempered joint. Tempering each conditional and sampling is not sampling from the tempered joint p(x_1..x_T)^{1/T}, which would require renormalizing over all sequences. The two differ because the per-step normalizers Z_t do not cancel across steps. A consequence: a sequence of greedy choices is not the most probable sequence, which is why beam search exists.
- Truncation is a different operation. Top-k and top-p sampling set probabilities to zero and renormalize; that is truncation, not a power transform, and is usually combined with temperature.
- Calibration, not uncertainty. The model's p_θ is not a posterior; sharpening it with T < 1 sharpens its miscalibration too. The same knob is also used in reverse at evaluation, fitting T on a validation set to calibrate confidences (temperature scaling), where it is a one-parameter recalibration rather than annealing.

---

## 8. DPO = Bradley Terry MLE

📊 **Statistics:** Bradley Terry (1952) models pairwise comparisons. Each item i has a score s_i and the probability that i beats j is the logistic function of the score difference. Given observed comparisons, the scores are fit by maximum likelihood, which is logistic regression on difference features. With a parametric score s = r_φ(x, y) the MLE is:

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) ),    L(φ) = −Σ log σ( r_φ(x, y_w) − r_φ(x, y_l) )
```

🧠 **Deep learning:** RLHF first fits a reward model r_φ by exactly that loss on human preference pairs, then maximizes expected reward under a KL penalty to a reference policy, max_π E[r] − β KL(π ‖ π_ref). DPO skips the reward model and trains the policy directly on the pairs:

```
L_DPO(θ) = −Σ log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) )
```

🔗 **Why equal:** the KL-regularized objective has a closed-form optimum, π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x). Solve for the reward: r(x, y) = β log π*(y|x)/π_ref(y|x) + β log Z(x). Substitute into Bradley Terry. The log Z(x) terms cancel in the difference because both completions share the prompt. What remains is the DPO loss. So DPO is Bradley Terry MLE where the score function has been reparameterized through the policy, and fitting the policy fits the reward model and solves the RL problem in one step.

⚠️ **Where it breaks:** the identity is exact at the optimum, and the practical gaps are around it.

- Off-policy. The substitution assumes π_θ is the exact optimum of the RL problem for every y. In practice it is an MLE on a finite set of pairs, so it only constrains the policy on completions in the dataset. PPO-style RLHF samples from the current policy and sees its own mistakes; DPO never does, and the policy can drift arbitrarily on unseen completions.
- Only differences are identified. Bradley Terry fixes scores up to a constant per prompt, so the loss can decrease by lowering the log probability of both the winner and the loser as long as the gap grows. That happens in practice (likelihood displacement) and is a direct consequence of fitting a model of differences.
- Separable data. Logistic-regression MLE diverges when the pairs are perfectly separable, pushing the log ratios to infinity. β sets the scale but is not a regularizer against this; early stopping does that job.
- The preference model itself. Bradley Terry assumes a scalar score with transitive, annotator-independent preferences. Human preferences are intransitive and vary by annotator, and anything built on a scalar reward inherits that misspecification, DPO and RLHF alike.

---

## 9. GRPO = REINFORCE with group baseline

📊 **Statistics:** the score-function (likelihood-ratio) gradient estimator. The gradient of an expected reward under a parametric distribution can be written as an expectation of reward times the gradient of the log density, because ∇π = π ∇log π. Any baseline b that does not depend on the sample can be subtracted without changing the expectation, since E[∇log π] = 0, and a good baseline reduces variance. This is a control variate. REINFORCE (Williams, 1992) is the Monte Carlo version with sampled y.

```
∇_θ E_{y∼π_θ}[R(y)] = E_{y∼π_θ}[ (R(y) − b) ∇_θ log π_θ(y) ]
```

🧠 **Deep learning:** for a prompt x, GRPO samples G completions from the current policy, scores them with a reward r_i, and forms an advantage by standardizing within the group. The update uses a PPO-style clipped probability ratio ρ and a KL penalty to a reference model, with the advantage applied to every token of completion i.

```
A_i = (r_i − mean_j r_j) / std_j r_j
L = (1/G) Σ_i (1/|y_i|) Σ_t min( ρ_{i,t} A_i, clip(ρ_{i,t}, 1−ε, 1+ε) A_i ) − β KL(π_θ ‖ π_ref)
```

🔗 **Why equal:** at the first gradient step the ratio ρ equals one, the clip is inactive, and the gradient of the objective is Σ_i A_i ∇log π_θ(y_i), which is REINFORCE with baseline b = group mean reward. That mean is a Monte Carlo estimate of the expected reward for prompt x, the value V(x), so the group plays the role of the critic network that PPO would train to estimate the baseline. GRPO replaces a learned baseline with a sampled one, which is why it needs no value model.

⚠️ **Where it breaks:** each extra ingredient moves it away from the plain estimator.

- The baseline is not independent of the sample. The group mean includes r_i itself, so the control variate depends on the action it is subtracting from, which biases the estimator by a term of order 1/G. Using the mean of the other G − 1 samples (leave-one-out) restores unbiasedness.
- Division by the standard deviation. Baselines are additive; a multiplicative rescaling is not a baseline. It equalizes the gradient magnitude across prompts regardless of how spread their rewards are, which changes the objective being optimized. When all G rewards are equal it gives 0/0, so prompts the model always gets right or always gets wrong contribute nothing.
- Clipping and KL. After the first step, ρ ≠ 1 and the clipped surrogate is a trust-region approximation, not the policy gradient. The KL term adds a penalty REINFORCE does not have.
- Per-token normalization. Dividing by the length |y_i| reweights sequences relative to plain REINFORCE on the sequence log probability, and the sequence-level reward is spread uniformly over tokens, which is a crude credit assignment.

---

## 10. Scaling laws = power-law regression

📊 **Statistics:** a power law y = a x^{−b} is a straight line in log-log coordinates, so the exponent is the slope of an ordinary least-squares fit on logged data. With an additive asymptote the relation is no longer linear in logs and is fit by nonlinear least squares, with E the limit as x → ∞.

```
log y = log a − b log x        or        y = E + a x^{−b}
```

🧠 **Deep learning:** Kaplan et al. (2020) fit loss against parameters N, tokens D and compute C separately as pure power laws. Hoffmann et al. (2022, "Chinchilla") fit a joint form on hundreds of training runs and then choose N and D to minimize it under the compute constraint C ≈ 6ND, obtaining the rule that tokens should scale in proportion to parameters.

```
L(N, D) = E + A / N^α + B / D^β,     minimize subject to 6ND = C
```

🔗 **Why equal:** the law is a regression model. The data points are (N, D, final loss) triples from training runs, the exponents and coefficients are the fitted parameters, and the compute-optimal rule is a Lagrange-multiplier calculation on the fitted surface. The irreducible term E is the entropy floor from item 2, the loss that no model of any size can go below because the data itself is random. Nothing in the procedure is specific to neural networks.

⚠️ **Where it breaks:** a fit is only as good as its data and its range.

- Extrapolation. The form is fitted on small runs and used to plan runs orders of magnitude larger. No mechanism guarantees a power law holds there; the functional form is a choice, and alternatives fit the same data about as well. The practice is empirical, the theories are post hoc.
- Confounds. Kaplan and Chinchilla disagree on the exponents partly because Kaplan used a fixed learning-rate schedule that was not tuned per run length. The coefficients depend on the architecture, tokenizer, data mix, and schedule, so they are not universal constants, and repeating data breaks the D term.
- Loss is not capability. The regression predicts cross-entropy. Downstream task scores are not smooth functions of cross-entropy, which is what the "emergence" debate is about.
- Regression assumptions. Runs are not independent noisy draws around a curve: they share hyperparameters and seeds, and the loss depends on where in the schedule you measure. Standard errors from the fit are not meaningful, and the goodness of fit does not transfer to the extrapolation region.
