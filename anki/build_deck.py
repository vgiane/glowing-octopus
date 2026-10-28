"""Build the Anki deck for the Statistics to Deep Learning Map note.

Two note types:
- term cards: one per glossary bullet in the note, with the definition from the
  note and, where the definition is essentially an equation, that equation set
  as a MathJax display block (kept here in LaTeX; the note uses Unicode).
- equation cards: one per numbered section. Back shows the deep-learning side
  first (left of the equals sign), then the statistics side, each with a one
  line gloss and its equation, then a short "where it breaks" list.

Run from the repo root:
    uv run --group anki python anki/build_deck.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import genanki

ROOT = Path(__file__).resolve().parent.parent
NOTE = ROOT / "Statistics to Deep Learning Map.md"
OUT = Path(__file__).resolve().parent / "statistics_to_deep_learning_map.apkg"

# Fixed ids so re-importing lands in the same deck and note types. The model
# ids were bumped when the field layout changed (October 2026); Anki cannot
# update a note type whose fields changed, so a new id is the clean path.
DECK_ID = 1_729_384_650
TERM_MODEL_ID = 1_729_384_653
EQ_MODEL_ID = 1_729_384_654

# Note GUIDs are derived from a stable key (the glossary term, or the section
# key below), never from the fields or the model id. Editing a definition,
# equation or gloss therefore updates the existing card on re-import instead
# of creating a duplicate. Renaming a term or key makes a new card.
GUID_NS = "stats-dl-map"

CSS = """
.card { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 20px;
        text-align: left; color: #222; background: #fff; padding: 12px; }
.front { font-weight: bold; }
.group { font-size: 14px; color: #777; margin-top: 10px; }
hr#answer { margin: 14px 0; }
.side { margin: 10px 0 16px 0; }
.label { font-weight: bold; }
.gloss { color: #444; }
.extra { margin: 6px 0 16px 0; font-size: 17px; color: #444; }
.breaks ul { margin: 6px 0 0 0; padding-left: 22px; font-size: 17px; }
.breaks li { margin-bottom: 4px; }
.MathJax_Display, mjx-container[display="true"] { margin: 8px 0 !important; }
"""

term_model = genanki.Model(
    TERM_MODEL_ID,
    "Stats-DL term",
    fields=[{"name": "Term"}, {"name": "Definition"}, {"name": "Equation"}, {"name": "Group"}],
    templates=[{
        "name": "Term -> Definition",
        "qfmt": '<div class="front">{{Term}}</div><div class="group">{{Group}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer">{{Definition}}'
                '{{#Equation}}<div class="side">{{Equation}}</div>{{/Equation}}',
    }],
    css=CSS,
)

eq_model = genanki.Model(
    EQ_MODEL_ID,
    "Stats-DL equation",
    fields=[
        {"name": "Title"}, {"name": "Hint"},
        {"name": "LeftName"}, {"name": "LeftGloss"}, {"name": "LeftEq"},
        {"name": "RightName"}, {"name": "RightGloss"}, {"name": "RightEq"},
        {"name": "Extra"}, {"name": "Breaks"},
    ],
    templates=[{
        "name": "Title -> Equation",
        "qfmt": '<div class="front">{{Title}}</div>'
                '<div class="group">Write the equation(s) that make the two sides equal.</div>'
                '<div class="group">{{Hint}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer">'
                '<div class="side"><span class="label">🧠 {{LeftName}}</span> '
                '<span class="gloss">{{LeftGloss}}</span>{{LeftEq}}</div>'
                '<div class="side"><span class="label">📊 {{RightName}}</span> '
                '<span class="gloss">{{RightGloss}}</span>{{RightEq}}</div>'
                '{{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}'
                '<div class="breaks"><span class="label">⚠️ Where it breaks</span>{{Breaks}}</div>',
    }],
    css=CSS,
)


def tex(s: str) -> str:
    """Fields are HTML: a bare '<' as in x_{<t} opens a tag and breaks the card.
    Escaping is safe because MathJax reads the parsed text, where &lt; is '<' again."""
    return html.escape(s, quote=False)


# --------------------------------------------------------------------------
# Glossary cards
# --------------------------------------------------------------------------

# Display equation for a term, keyed by the term exactly as written in the note.
GLOSSARY_EQUATIONS: dict[str, str] = {
    # Deep-learning side
    "LM": r"\[ p_\theta(x_t \mid x_{<t}) = \operatorname{softmax}\big(f_\theta(x_{<t})\big)_{x_t} \]",
    "Cross-entropy": r"\[ \mathrm{CE}(q, p) = -\sum_v q(v) \log p(v) \]",
    "Perplexity": r"\[ \mathrm{PPL} = \exp\!\left( -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \right) \]",
    "SGD": r"\[ \hat g = \frac{1}{B} \sum_{i=1}^{B} \nabla \ell(\theta; x_i), \qquad \theta \leftarrow \theta - \eta\, \hat g \]",
    "Adam": r"\[ m \leftarrow \beta_1 m + (1-\beta_1)\, g, \quad v \leftarrow \beta_2 v + (1-\beta_2)\, g^2, "
            r"\quad \theta \leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon} \]",
    "AdamW": r"\[ \theta \leftarrow (1 - \eta\lambda)\, \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon} \]",
    "Weight decay": r"\[ \theta \leftarrow (1 - \eta\lambda)\, \theta - \eta\, g \]",
    "Temperature": r"\[ p_T(v) = \operatorname{softmax}(z / T)_v = \frac{\exp(z_v / T)}{\sum_u \exp(z_u / T)} \]",
    "DPO": r"\[ \mathcal{L}_{\mathrm{DPO}}(\theta) = -\sum \log \sigma\!\left( "
           r"\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} "
           r"- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \]",
    "GRPO": r"\[ A_i = \frac{r_i - \operatorname{mean}_j r_j}{\operatorname{std}_j r_j} \]",
    "Scaling laws": r"\[ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} \]",
    # Statistics side
    "Autoregressive conditional density": r"\[ p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1}) \]",
    "NLL": r"\[ \mathrm{NLL}(\theta) = -\log p_\theta(D), \qquad \hat\theta_{\mathrm{MLE}} = \arg\min_\theta \mathrm{NLL}(\theta) \]",
    "Robbins Monro": r"\[ \theta_{t+1} = \theta_t - a_t\, G(\theta_t, \xi_t), \qquad \sum_t a_t = \infty, \qquad \sum_t a_t^2 < \infty \]",
    "Preconditioned SGD": r"\[ \theta \leftarrow \theta - \eta\, P^{-1} g, \qquad \text{diagonal: } \theta_i \leftarrow \theta_i - \frac{\eta}{d_i}\, g_i \]",
    "Gaussian prior": r"\[ \theta \sim \mathcal{N}(0, \sigma^2 I), \qquad -\log p(\theta) = \frac{\lVert \theta \rVert^2}{2\sigma^2} + \text{const} \]",
    "MAP": r"\[ \theta_{\mathrm{MAP}} = \arg\max_\theta\; p(D \mid \theta)\, p(\theta) = \arg\min_\theta\; \mathrm{NLL}(\theta) - \log p(\theta) \]",
    "Tempering": r"\[ p_T(x) \propto p(x)^{1/T} = \exp\!\left(-\frac{E(x)}{T}\right), \qquad E(x) = -\log p(x) \]",
    "Bradley Terry": r"\[ P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1 + e^{-(s_i - s_j)}} \]",
    "MLE": r"\[ \hat\theta_{\mathrm{MLE}} = \arg\max_\theta\; p(D \mid \theta) \]",
    "REINFORCE": r"\[ \nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta}[R(y)] = \mathbb{E}_{y \sim \pi_\theta}\big[ R(y)\, \nabla_\theta \log \pi_\theta(y) \big] \]",
    "Baseline": r"\[ \mathbb{E}_{y \sim \pi_\theta}\big[ (R(y) - b)\, \nabla_\theta \log \pi_\theta(y) \big], "
                r"\qquad \mathbb{E}_{y \sim \pi_\theta}\big[ b\, \nabla_\theta \log \pi_\theta(y) \big] = 0 \]",
    "Power-law regression": r"\[ y = a\, x^{b} \quad \Longleftrightarrow \quad \log y = \log a + b \log x \]",
    # Other
    "KL": r"\[ \mathrm{KL}(q \,\|\, p) = \sum_v q(v) \log \frac{q(v)}{p(v)} \;\ge\; 0 \]",
    "RLHF": r"\[ \max_\pi\; \mathbb{E}_{y \sim \pi}[r(x, y)] - \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\mathrm{ref}}) \]",
    "PPO": r"\[ \mathbb{E}\big[ \min( \rho A,\; \operatorname{clip}(\rho, 1-\epsilon, 1+\epsilon)\, A ) \big], "
           r"\qquad \rho = \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{old}}(y \mid x)} \]",
    # Prerequisites
    "Logits": r"\[ z = f_\theta(x_{<t}) \in \mathbb{R}^{|V|} \]",
    "Softmax": r"\[ p(v) = \frac{\exp(z_v)}{\sum_u \exp(z_u)} \]",
    "Partition function Z": r"\[ Z = \sum_u \exp(z_u) \]",
    "One-hot target": r"\[ q(v) = \mathbf{1}[v = x_t] \]",
    "Entropy H(q)": r"\[ H(q) = -\sum_v q(v) \log q(v) \]",
    "Unbiased estimator": r"\[ \mathbb{E}[\hat g] = \nabla L(\theta) \]",
    "Hessian": r"\[ H = \nabla^2 L(\theta), \qquad \theta \leftarrow \theta - H^{-1} \nabla L(\theta) \]",
    "Fisher information F": r"\[ F = \mathbb{E}\big[ \nabla \log p\, \nabla \log p^{\top} \big] = -\mathbb{E}\big[ \nabla^2 \log p \big] \]",
    "Prior, likelihood, posterior": r"\[ p(\theta \mid D) \propto p(D \mid \theta)\, p(\theta) \]",
    "Policy π_θ(y | x)": r"\[ \pi_\theta(y \mid x), \qquad \pi_{\mathrm{ref}}(y \mid x), \qquad r(x, y) \in \mathbb{R} \]",
    "Score function ∇_θ log π_θ(y)": r"\[ \mathbb{E}_{y \sim \pi_\theta}\big[ \nabla_\theta \log \pi_\theta(y) \big] = 0 \]",
    "Control variate": r"\[ \mathbb{E}\big[ f(y) - c\, h(y) \big] = \mathbb{E}[f(y)] \quad \text{when } \mathbb{E}[h(y)] = 0 \]",
    "Least squares": r"\[ \hat\theta = \arg\min_\theta \sum_i \big( y_i - f(x_i; \theta) \big)^2 \]",
}

# Short card labels for the note's glossary headings; unknown headings pass through.
GROUP_NAMES: dict[str, str] = {
    "Deep-learning side (left of each equals sign)": "Deep learning side (left)",
    "Statistics side (right of each equals sign)": "Statistics side (right)",
    "Prerequisites used in the derivations": "Prerequisite",
}

# Front-side names for terms whose name in the note carries Unicode math.
# Inline MathJax so the front matches the typeset back. GUIDs still use the
# note's term, so changing a display name does not create a new card.
GLOSSARY_NAMES: dict[str, str] = {
    "Entropy H(q)": r"Entropy \(H(q)\)",
    "Fisher information F": r"Fisher information \(F\)",
    "Partition function Z": r"Partition function \(Z\)",
    "Policy π_θ(y | x)": r"Policy \(\pi_\theta(y \mid x)\)",
    "Score function ∇_θ log π_θ(y)": r"Score function \(\nabla_\theta \log \pi_\theta(y)\)",
}

# Definitions whose text in the note carries the formula inline. The card uses
# this prose instead and shows the formula as the display equation above.
GLOSSARY_TEXT: dict[str, str] = {
    "Softmax": "turns logits into a probability vector. Exponentiate, then divide by the sum "
               "so the entries are positive and sum to one.",
    "Partition function Z": "the normalizing constant in the softmax, or more generally the sum "
                            "or integral that makes an unnormalized distribution sum to one.",
    "One-hot target": "the observed token written as a probability vector with 1 at that token "
                      "and 0 elsewhere; the Kronecker delta seen as a distribution. All mass on one outcome.",
    "Entropy H(q)": "the expected surprise under q; the lower bound of cross-entropy against q.",
    "Fisher information F": "the expected outer product of the score. Equals the expected Hessian "
                            "of the NLL under the model, so it is a curvature matrix defined from "
                            "the likelihood alone. Natural gradient divides the gradient by it.",
    "Prior, likelihood, posterior": "what you believe about the parameters before data, the "
                                    "likelihood of the data given them, and the posterior as the "
                                    "update. MAP maximizes the posterior; MLE maximizes only the likelihood.",
    "Policy π_θ(y | x)": "the LLM viewed as a distribution over completions y given prompt x. "
                         "The reference policy is the frozen starting model. The reward is a "
                         "scalar score for a completion.",
    "Score function ∇_θ log π_θ(y)": "the gradient of the log probability of a sample. Its "
                                     "expectation under the policy is zero, which is what makes baselines free.",
    "Power-law regression": "fitting a power law, which is a straight line in log-log coordinates.",
    "Logits": "the raw real-valued outputs of the network, one per vocabulary entry, before any normalization.",
}


def glossary_cards(text: str) -> list[genanki.Note]:
    """Parse '- Term: definition' bullets between the glossary heading and the first rule."""
    start = text.index("## Abbreviations and named terms")
    end = text.index("\n---\n", start)
    group = ""
    notes = []
    seen = set()
    for line in text[start:end].splitlines():
        if line.endswith(":") and not line.startswith("-"):
            group = GROUP_NAMES.get(line.rstrip(":"), line.rstrip(":"))
            continue
        m = re.match(r"^- (.+?): (.+)$", line)
        if not m:
            continue
        term, definition = m.group(1), m.group(2)
        seen.add(term)
        definition = GLOSSARY_TEXT.get(term, definition)
        equation = GLOSSARY_EQUATIONS.get(term, "")
        notes.append(genanki.Note(
            model=term_model,
            fields=[tex(GLOSSARY_NAMES.get(term, term)), html.escape(definition, quote=False),
                    tex(equation), group],
            guid=genanki.guid_for(GUID_NS, "term", term),
            tags=["stats-dl-map", "glossary"],
        ))
    missing = (set(GLOSSARY_EQUATIONS) | set(GLOSSARY_TEXT) | set(GLOSSARY_NAMES)) - seen
    if missing:
        sys.exit(f"glossary keys not found in the note: {sorted(missing)}")
    return notes


# --------------------------------------------------------------------------
# Equation cards
# --------------------------------------------------------------------------

# key: stable GUID key; title: front; hint: front, DL side first;
# left: (name, gloss, equation) for the deep-learning side;
# right: (name, gloss, equation) for the statistics side;
# extra: optional bridging equation, no prose; breaks: short list.
EQUATIONS = [
    dict(
        key="lm",
        title="LM = autoregressive conditional density",
        hint="The network as one factor, and the chain rule.",
        left=("LM",
              "A network maps the prefix to logits; softmax gives a categorical over the next token.",
              r"\[ p_\theta(x_t \mid x_{<t}) = \operatorname{softmax}\big(f_\theta(x_{<t})\big)_{x_t} \]"),
        right=("Autoregressive conditional density",
               "Chain rule of probability, true for any joint; each factor conditions on the whole past.",
               r"\[ p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1}) \]"),
        extra="",
        breaks=[
            "Finite context window: a k-th order Markov model, a smaller family than the chain rule.",
            "Fixed left-to-right order: 'A is B' does not give 'B is A'; no infilling without another model.",
        ],
    ),
    dict(
        key="cross-entropy",
        title="Cross-entropy = NLL",
        hint="The one-hot collapse, the NLL of a sequence, and the entropy plus KL decomposition.",
        left=("Cross-entropy",
              "Loss between a target q and the softmax output; the target is one-hot on the next token.",
              r"\[ \mathrm{CE}\big(\mathrm{onehot}(x_t), p_\theta\big) = -\sum_{v} \mathbf{1}[v = x_t] \log p_\theta(v \mid x_{<t}) "
              r"= -\log p_\theta(x_t \mid x_{<t}) \]"),
        right=("NLL",
               "Minus the log probability of the data, a sum over positions; minimizing it is MLE.",
               r"\[ \mathrm{NLL}(\theta) = -\log p_\theta(x_1, \ldots, x_T) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \]"),
        extra=r"In expectation over the data: \[ \mathbb{E}_q[-\log p_\theta] = H(q) + \mathrm{KL}(q \,\|\, p_\theta) \]",
        breaks=[
            "Label smoothing: extra terms survive the sum, no longer the likelihood of any data.",
            "Distillation: KL to the teacher, not to the data.",
            "Non-i.i.d. tokens: still the MLE point estimate, but i.i.d. asymptotics do not apply.",
        ],
    ),
    dict(
        key="perplexity",
        title="Perplexity = exp(mean NLL)",
        hint="Mean per-token loss through exp, and the geometric mean form.",
        left=("Perplexity",
              "Mean cross-entropy per token on held-out text, exponentiated. In bits, 2 to that mean.",
              r"\[ \mathrm{PPL} = \exp\!\left( -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \right) \]"),
        right=("exp(mean NLL)",
               "exp of an entropy is the number of equally likely outcomes with that entropy; on data, the geometric mean of 1/p.",
               r"\[ \mathrm{PPL} = \left( \prod_{t=1}^{T} \frac{1}{p_\theta(x_t \mid x_{<t})} \right)^{1/T}, \qquad \mathrm{PPL}(q) = e^{H(q)} \]"),
        extra="",
        breaks=[
            "Tokenizer dependence: per-token NLL is not invariant; compare per character or per byte.",
            "Nonlinear scale: average the NLL first, then exponentiate.",
            "Not a measure of ability: it scores density estimation on the test text.",
        ],
    ),
    dict(
        key="sgd",
        title="SGD = Robbins Monro",
        hint="The minibatch gradient and its step, then the stochastic approximation update with its two step-size conditions.",
        left=("SGD",
              "The loss is an expectation over data; a minibatch gives an unbiased gradient; step against it.",
              r"\[ \hat g = \frac{1}{B} \sum_{i=1}^{B} \nabla \ell(\theta; x_i), \qquad \mathbb{E}[\hat g] = \nabla L(\theta), "
              r"\qquad \theta \leftarrow \theta - \eta_t\, \hat g \]"),
        right=("Robbins Monro",
               "Find the root of g(θ) = E[G(θ, ξ)] from noisy evaluations, with steps that sum to infinity but square-sum to a finite number.",
               r"\[ \theta_{t+1} = \theta_t - a_t\, G(\theta_t, \xi_t), \qquad \sum_t a_t = \infty, \qquad \sum_t a_t^2 < \infty \]"),
        extra="",
        breaks=[
            "Nonconvexity: convergence only to some stationary point.",
            "Finite-horizon schedules (warmup, cosine): not covered by the theorem.",
            "Momentum, Adam, clipping, weight decay: a different update, each with its own analysis.",
        ],
    ),
    dict(
        key="adam",
        title="Adam = diagonally preconditioned SGD",
        hint="The Adam update, then the preconditioned step and its diagonal case.",
        left=("Adam",
              "Two running averages per parameter, bias-corrected; divide one by the square root of the other.",
              r"\[ m \leftarrow \beta_1 m + (1-\beta_1)\, g, \qquad v \leftarrow \beta_2 v + (1-\beta_2)\, g^2, "
              r"\qquad \theta \leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon} \]"),
        right=("Diagonally preconditioned SGD",
               "Multiply the gradient by the inverse of a positive definite matrix (Hessian for Newton, Fisher for natural gradient); diagonal keeps one scale per coordinate.",
               r"\[ \theta \leftarrow \theta - \eta\, P^{-1} g, \qquad \theta_i \leftarrow \theta_i - \frac{\eta}{d_i}\, g_i \]"),
        extra="",
        breaks=[
            "The square root: divides by √F, not F; no curvature derivation, not second order.",
            "Empirical Fisher: observed labels, and v also contains gradient noise.",
            "Diagonal only: correlations inside a weight matrix are discarded.",
            "No fixed metric: v changes every step; can fail to converge where SGD does (AMSGrad fixes).",
        ],
    ),
    dict(
        key="weight-decay",
        title="Weight decay = Gaussian prior (approximate under AdamW)",
        hint="The weight-decay step, then the MAP objective with an isotropic Gaussian prior.",
        left=("Weight decay",
              "Shrink every weight toward zero by a factor (1 − ηλ) each step, on top of the gradient step.",
              r"\[ \theta \leftarrow \theta - \eta\, (g + \lambda \theta) = (1 - \eta\lambda)\, \theta - \eta\, g \]"),
        right=("Gaussian prior",
               "MAP maximizes log likelihood plus log prior; an isotropic Gaussian prior adds an L2 penalty (ridge).",
               r"\[ \theta_{\mathrm{MAP}} = \arg\min_\theta\; \mathrm{NLL}(\theta) + \frac{\lambda}{2} \lVert \theta \rVert^2, "
               r"\qquad \lambda = \frac{1}{\sigma^2} \]"),
        extra="",
        breaks=[
            "Adam with L2 in the loss: the penalty gets preconditioned, the prior is no longer isotropic.",
            "AdamW: decay outside the preconditioner; no objective, no σ², the prior is an analogy.",
            "Normalization layers: scale-invariant weights, decay only raises the effective learning rate.",
            "A prior on weights is not the prior on functions you would choose.",
        ],
    ),
    dict(
        key="temperature",
        title="Temperature = tempering",
        hint="Softmax with temperature, then the tempered distribution.",
        left=("Temperature",
              "Divide the logits by T before the softmax; T = 1 is the model, T → 0 is greedy.",
              r"\[ p_T(v) = \operatorname{softmax}(z / T)_v = \frac{\exp(z_v / T)}{\sum_u \exp(z_u / T)} \]"),
        right=("Tempering",
               "Raise a distribution to the power 1/T and renormalize; the Boltzmann distribution at temperature T.",
               r"\[ p_T(x) \propto p(x)^{1/T} = \exp\!\left(-\frac{E(x)}{T}\right), \qquad E(x) = -\log p(x) \]"),
        extra="",
        breaks=[
            "Per token, not the tempered joint: greedy is not the most probable sequence.",
            "Top-k and top-p are truncation, not a power transform.",
            "Sharpening with T < 1 also sharpens miscalibration; p_θ is not a posterior.",
        ],
    ),
    dict(
        key="dpo",
        title="DPO = Bradley Terry MLE",
        hint="The DPO loss, then the Bradley Terry likelihood and its MLE.",
        left=("DPO",
              "Train the policy directly on preference pairs, against a frozen reference, with scale β.",
              r"\[ \mathcal{L}_{\mathrm{DPO}}(\theta) = -\sum \log \sigma\!\left( "
              r"\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} "
              r"- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \]"),
        right=("Bradley Terry MLE",
               "P(winner beats loser) is the sigmoid of the score difference; fit the scores by maximum likelihood.",
               r"\[ P(y_w \succ y_l \mid x) = \sigma\big( r(x, y_w) - r(x, y_l) \big), "
               r"\qquad \mathcal{L}(\phi) = -\sum \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \]"),
        extra=r"Reward through the KL-regularized optimum: "
              r"\[ r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x) \]",
        breaks=[
            "Off-policy: constrains the policy only on dataset completions; it never sees its own samples.",
            "Only differences identified: both log probabilities can fall while the gap grows.",
            "Separable pairs: the MLE diverges; β is not a regularizer, early stopping is.",
            "Scalar, transitive, annotator-independent preferences are assumed.",
        ],
    ),
    dict(
        key="grpo",
        title="GRPO = REINFORCE with group baseline",
        hint="The group advantage and the GRPO objective, then REINFORCE with a baseline.",
        left=("GRPO",
              "Sample G completions per prompt, standardize their rewards within the group, PPO clip plus KL to a reference, same advantage on every token.",
              r"\[ A_i = \frac{r_i - \operatorname{mean}_j r_j}{\operatorname{std}_j r_j} \]"
              r"\[ \mathcal{L} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t} "
              r"\min\!\Big( \rho_{i,t} A_i,\; \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\, A_i \Big) "
              r"- \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}) \]"),
        right=("REINFORCE with baseline",
               "Score-function gradient estimator; a baseline independent of the sample leaves the expectation unchanged and lowers variance.",
               r"\[ \nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta}[R(y)] = "
               r"\mathbb{E}_{y \sim \pi_\theta}\big[ (R(y) - b)\, \nabla_\theta \log \pi_\theta(y) \big] \]"),
        extra=r"At the first step, ρ = 1: "
              r"\[ \nabla_\theta \mathcal{L} = \sum_i A_i\, \nabla_\theta \log \pi_\theta(y_i), \qquad b = \operatorname{mean}_j r_j \approx V(x) \]",
        breaks=[
            "The group mean includes r_i: bias of order 1/G; leave-one-out restores unbiasedness.",
            "Division by the std is not a baseline; 0/0 when all rewards agree.",
            "After step one, clipping and KL make it a trust-region surrogate.",
            "Per-token length normalization reweights sequences; uniform credit assignment.",
        ],
    ),
    dict(
        key="scaling-laws",
        title="Scaling laws = power-law regression",
        hint="The Chinchilla fit with its constraint, then the power law in log-log form.",
        left=("Scaling laws",
              "Fit loss against parameters N and tokens D on many runs, then pick N, D minimizing it under the compute constraint.",
              r"\[ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}, \qquad \text{minimize subject to } 6ND = C \]"),
        right=("Power-law regression",
               "A power law is a line in log-log coordinates, slope by ordinary least squares; with an asymptote E it is nonlinear least squares.",
               r"\[ \log y = \log a - b \log x \qquad \text{or} \qquad y = E + a\, x^{-b} \]"),
        extra="",
        breaks=[
            "Extrapolation orders of magnitude past the fitted range; the form is a choice.",
            "Confounds: schedule, tokenizer, data mix; repeated data breaks the D term.",
            "Predicts cross-entropy, not downstream capability.",
            "Runs are not independent draws; standard errors are not meaningful.",
        ],
    ),
]


def equation_cards() -> list[genanki.Note]:
    notes = []
    for e in EQUATIONS:
        lname, lgloss, leq = e["left"]
        rname, rgloss, req = e["right"]
        breaks = "<ul>" + "".join(f"<li>{html.escape(b, quote=False)}</li>" for b in e["breaks"]) + "</ul>"
        notes.append(genanki.Note(
            model=eq_model,
            fields=[e["title"], e["hint"],
                    lname, html.escape(lgloss, quote=False), tex(leq),
                    rname, html.escape(rgloss, quote=False), tex(req),
                    tex(e["extra"]), breaks],
            guid=genanki.guid_for(GUID_NS, "equation", e["key"]),
            tags=["stats-dl-map", "equation"],
        ))
    return notes


def main() -> None:
    text = NOTE.read_text(encoding="utf-8")
    deck = genanki.Deck(DECK_ID, "Statistics to Deep Learning Map")
    terms = glossary_cards(text)
    eqs = equation_cards()
    for note in terms + eqs:
        deck.add_note(note)
    genanki.Package(deck).write_to_file(str(OUT))
    print(f"wrote {OUT.name}: {len(terms)} term cards, {len(eqs)} equation cards")


if __name__ == "__main__":
    main()
