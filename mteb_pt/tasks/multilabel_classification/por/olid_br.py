"""OlidBrMultilabelClassification — multi-label offensive-type identification (PT-BR).

OLID-BR (dougtrajano/olid-br): native PT-BR social-media texts (YouTube +
Twitter) annotated with an 11-type offense taxonomy (health, ideology, insult,
lgbtqphobia, other_lifestyle, physical_aspects, profanity_obscene, racism,
religious_intolerance, sexism, xenophobia). Multi-label — richer than HateBR's
binary hate label. Non-offensive texts carry an empty label set.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification

_REPO = "dougtrajano/olid-br"
_REVISION = "84b0d7dd4309be677a47c535632a9398ff1897bd"

_TYPES = [
    "health",
    "ideology",
    "insult",
    "lgbtqphobia",
    "other_lifestyle",
    "physical_aspects",
    "profanity_obscene",
    "racism",
    "religious_intolerance",
    "sexism",
    "xenophobia",
]


class OlidBrMultilabelClassification(AbsTaskMultilabelClassification):
    """Multi-label offensive-type identification (11 types) on native PT-BR texts."""

    metadata = TaskMetadata(
        name="OlidBrMultilabelClassification",
        description=(
            "Multi-label offensive-language type identification on native "
            "Brazilian-Portuguese social-media texts (OLID-BR). Each text is "
            "labelled with the subset of 11 offense types present (insult, "
            "profanity, racism, sexism, lgbtqphobia, xenophobia, religious "
            "intolerance, etc.); non-offensive texts have no labels. Richer than "
            "the binary HateBR hate-speech task."
        ),
        reference="https://huggingface.co/datasets/dougtrajano/olid-br",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2022-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{olid_br,
    title        = {{OLID-BR}: Offensive Language Identification Dataset for {B}razilian {P}ortuguese},
    author       = {Trajano, Douglas and Bordini, Rafael H. and Vieira, Renata},
    howpublished = {HuggingFace dataset \texttt{dougtrajano/olid-br}},
    url          = {https://huggingface.co/datasets/dougtrajano/olid-br},
}""",
    )

    input_column_name = "text"
    label_column_name = "label"

    def dataset_transform(self, **kwargs) -> None:
        """Build the multi-label set from the 11 boolean offense-type columns."""
        self.dataset = self.dataset.map(
            lambda ex: {"label": [t for t in _TYPES if ex[t]]}
        ).select_columns(["text", "label"])
