# Anki decks

Decks generated from the notes in this repository. Each deck has a build script next to it; the `.apkg` is the output and is committed so it can be imported directly.

## Decks

| Deck | Source note | Build script | Cards |
| --- | --- | --- | --- |
| `statistics_to_deep_learning_map.apkg` | `Statistics to Deep Learning Map.md` | `build_deck.py` | 40 glossary terms, 10 equation sections |

## Build

From the repo root:

```sh
uv sync --group anki
uv run --group anki python anki/build_deck.py
```

The script reads the glossary bullets from the note, so editing a definition in the note and rebuilding updates the card. All equations are written in LaTeX inside the script, because the note keeps them in Unicode for viewers without math support. A few glossary definitions carry their formula inline in the note; for those the script has its own prose (`GLOSSARY_TEXT`) and shows the formula as a display equation instead.

## Import

1. Open Anki, then File, Import, and pick the `.apkg`. Set "Update notes" to "Always".
2. Re-importing after a rebuild updates existing cards in place. Each note's GUID is derived from its term (glossary cards) or a fixed section key (equation cards), so edits to a definition, equation or gloss keep the same card and its review history. Renaming a term or key creates a new card. Deck id, note-type ids and the GUID scheme are fixed in the script; do not change them.
3. If you imported a build from before 28 October 2026, delete it first, otherwise every card is duplicated once: in the browser search `tag:stats-dl-map`, select all, delete the notes; then in Tools, Manage Note Types, delete the old `Stats-DL term` and `Stats-DL equation` types, which are now empty. Older builds hashed GUIDs from the card text and used a different field layout.
4. Equations use MathJax delimiters `\[ ... \]`, which Anki desktop, AnkiMobile and AnkiDroid render without any add-on.

## Card types

- **Stats-DL term**: front is the term and its glossary group; back is the one-line definition followed by the defining equation as a display block, where the term has one.
- **Stats-DL equation**: front is the section title plus a hint of what to write. Back shows the deep-learning side first (left of the equals sign) and then the statistics side, each with a one-line gloss and its equation; then, for some sections, the bridging equation without commentary; then a short "where it breaks" list. The "why equal" derivation is left out on purpose.

Tags: every card carries `stats-dl-map`, plus `glossary` or `equation`.
