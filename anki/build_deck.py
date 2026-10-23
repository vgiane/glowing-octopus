"""Build the Anki deck for the Statistics to Deep Learning Map note.

Reads the glossary bullets from the note and adds one card per equation section.
Equations are written in MathJax, which Anki renders natively.

Run from the repo root:
    uv run --group anki python anki/build_deck.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import genanki

ROOT = Path(__file__).resolve().parent.parent
NOTE = ROOT / "Statistics to Deep Learning Map.md"
OUT = Path(__file__).resolve().parent / "statistics_to_deep_learning_map.apkg"

# Fixed ids so re-importing updates cards instead of duplicating them.
DECK_ID = 1_729_384_650
TERM_MODEL_ID = 1_729_384_651
EQ_MODEL_ID = 1_729_384_652

CSS = """
.card { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 20px;
        text-align: left; color: #222; background: #fff; padding: 12px; }
.front { font-weight: bold; }
.group { font-size: 14px; color: #777; margin-top: 10px; }
hr#answer { margin: 14px 0; }
"""

term_model = genanki.Model(
    TERM_MODEL_ID,
    "Stats-DL term",
    fields=[{"name": "Term"}, {"name": "Definition"}, {"name": "Group"}],
    templates=[{
        "name": "Term -> Definition",
        "qfmt": '<div class="front">{{Term}}</div><div class="group">{{Group}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer">{{Definition}}',
    }],
    css=CSS,
)

eq_model = genanki.Model(
    EQ_MODEL_ID,
    "Stats-DL equation",
    fields=[{"name": "Title"}, {"name": "Equation"}, {"name": "Hint"}],
    templates=[{
        "name": "Title -> Equation",
        "qfmt": '<div class="front">{{Title}}</div>'
                '<div class="group">Write the equation(s) that make the two sides equal.</div>'
                '<div class="group">{{Hint}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer">{{Equation}}',
    }],
    css=CSS,
)


def glossary_cards(text: str) -> list[genanki.Note]:
    """Parse '- Term: definition' bullets between the glossary heading and the first rule."""
    start = text.index("## Abbreviations and named terms")
    end = text.index("\n---\n", start)
    group = ""
    notes = []
    for line in text[start:end].splitlines():
        if line.endswith(":") and not line.startswith("-"):
            group = line.rstrip(":")
            continue
        m = re.match(r"^- (.+?): (.+)$", line)
        if m:
            term, definition = m.group(1), m.group(2)
            notes.append(genanki.Note(
                model=term_model,
                fields=[term, definition, group],
                tags=["stats-dl-map", "glossary"],
            ))
    return notes


EQUATIONS = [
    ("1. LM = autoregressive conditional density",
     r"\[ p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1}) \]"
     r"\[ p_\theta(x_t \mid x_{<t}) = \operatorname{softmax}\big(f_\theta(x_{<t})\big)_{x_t} \]",
     "Chain rule, and the network as one factor."),
    ("2. Cross-entropy = NLL",
     r"\[ \mathrm{NLL}(\theta) = -\log p_\theta(x_1, \ldots, x_T) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \]"
     r"\[ \mathrm{CE}\big(\mathrm{onehot}(x_t), p_\theta\big) = -\sum_{v} \mathbf{1}[v = x_t] \log p_\theta(v \mid x_{<t}) = -\log p_\theta(x_t \mid x_{<t}) \]"
     r"\[ \mathbb{E}_q[-\log p_\theta] = H(q) + \mathrm{KL}(q \,\|\, p_\theta) \]",
     "NLL of a sequence, the one-hot collapse, and the entropy plus KL decomposition."),
    ("3. Perplexity = exp(mean NLL)",
     r"\[ \mathrm{PPL} = \exp\!\left( -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \right) "
     r"= \left( \prod_{t=1}^{T} \frac{1}{p_\theta(x_t \mid x_{<t})} \right)^{1/T} \]",
     "Also as a geometric mean."),
    ("4. SGD = Robbins Monro",
     r"\[ \theta_{t+1} = \theta_t - a_t\, G(\theta_t, \xi_t), \qquad \sum_t a_t = \infty, \qquad \sum_t a_t^2 < \infty \]"
     r"\[ \hat g = \frac{1}{B} \sum_{i=1}^{B} \nabla \ell(\theta; x_i), \qquad \mathbb{E}[\hat g] = \nabla L(\theta), "
     r"\qquad \theta \leftarrow \theta - \eta_t\, \hat g \]",
     "Stochastic approximation update with its two step-size conditions, and the minibatch gradient."),
    ("5. Adam = diagonally preconditioned SGD",
     r"\[ \theta \leftarrow \theta - \eta\, P^{-1} g, \qquad \theta_i \leftarrow \theta_i - \frac{\eta}{d_i}\, g_i \]"
     r"\[ m \leftarrow \beta_1 m + (1-\beta_1)\, g, \qquad v \leftarrow \beta_2 v + (1-\beta_2)\, g^2, "
     r"\qquad \theta \leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon} \]",
     "Preconditioned step, diagonal case, and the Adam update."),
    ("6. Weight decay = Gaussian prior (approximate under AdamW)",
     r"\[ \theta_{\mathrm{MAP}} = \arg\min_\theta\; \mathrm{NLL}(\theta) + \frac{\lambda}{2} \lVert \theta \rVert^2, "
     r"\qquad \lambda = \frac{1}{\sigma^2} \]"
     r"\[ \theta \leftarrow \theta - \eta\, (g + \lambda \theta) = (1 - \eta\lambda)\, \theta - \eta\, g \]",
     "MAP objective with isotropic Gaussian prior, and the weight-decay step."),
    ("7. Temperature = tempering",
     r"\[ p_T(x) \propto p(x)^{1/T} = \exp\!\left(-\frac{E(x)}{T}\right), \qquad E(x) = -\log p(x) \]"
     r"\[ p_T(v) = \operatorname{softmax}(z / T)_v = \frac{\exp(z_v / T)}{\sum_u \exp(z_u / T)} \]",
     "Tempered distribution, and softmax with temperature."),
    ("8. DPO = Bradley Terry MLE",
     r"\[ P(y_w \succ y_l \mid x) = \sigma\big( r(x, y_w) - r(x, y_l) \big), "
     r"\qquad \mathcal{L}(\phi) = -\sum \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \]"
     r"\[ \mathcal{L}_{\mathrm{DPO}}(\theta) = -\sum \log \sigma\!\left( "
     r"\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} "
     r"- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \]",
     "Bradley Terry likelihood, and the DPO loss after substituting the closed-form reward."),
    ("9. GRPO = REINFORCE with group baseline",
     r"\[ \nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta}[R(y)] = "
     r"\mathbb{E}_{y \sim \pi_\theta}\big[ (R(y) - b)\, \nabla_\theta \log \pi_\theta(y) \big] \]"
     r"\[ A_i = \frac{r_i - \operatorname{mean}_j r_j}{\operatorname{std}_j r_j} \]"
     r"\[ \mathcal{L} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t} "
     r"\min\!\Big( \rho_{i,t} A_i,\; \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\, A_i \Big) "
     r"- \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}) \]",
     "REINFORCE with baseline, the group advantage, and the GRPO objective."),
    ("10. Scaling laws = power-law regression",
     r"\[ \log y = \log a - b \log x \qquad \text{or} \qquad y = E + a\, x^{-b} \]"
     r"\[ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}, \qquad \text{minimize subject to } 6ND = C \]",
     "Power law in log-log form, and the Chinchilla fit with its constraint."),
]


def equation_cards() -> list[genanki.Note]:
    # Fields are HTML: a bare '<' as in x_{<t} opens a tag and breaks the card.
    # Escaping is safe because MathJax reads the parsed text, where &lt; is '<' again.
    return [
        genanki.Note(model=eq_model, fields=[title, html.escape(eq, quote=False), hint],
                     tags=["stats-dl-map", "equation"])
        for title, eq, hint in EQUATIONS
    ]


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
