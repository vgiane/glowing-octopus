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

The script reads the glossary bullets from the note, so editing a definition in the note and rebuilding updates the card. The equations are written in LaTeX inside the script, because the note keeps them in Unicode for viewers without math support.

## Import

1. Open Anki, then File, Import, and pick the `.apkg`.
2. Re-importing after a rebuild updates existing cards in place. Deck and note-type ids are fixed in the script for that reason; do not change them.
3. Equations use MathJax delimiters `\[ ... \]`, which Anki desktop, AnkiMobile and AnkiDroid render without any add-on.

## Card types

- **Stats-DL term**: front is the term and its glossary group, back is the one-line definition.
- **Stats-DL equation**: front is the section title plus a hint of what to write, back is the typeset equation block.

Tags: every card carries `stats-dl-map`, plus `glossary` or `equation`.
