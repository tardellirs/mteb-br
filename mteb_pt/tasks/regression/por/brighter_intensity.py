"""BrighterEmotionIntensityRegression — total emotional intensity regression (PT-BR).

BRIGHTER dataset (SemEval-2025 Task 11) ptbr intensities split: each native
PT-BR text carries a 0-3 intensity for each of 6 emotions. The regression
target is the TOTAL emotional intensity (sum of the six, 0-18) — "how
emotionally charged is this text". Complements the AES essay-grade regression
with a social-media / affective target.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks.regression import AbsTaskRegression

_REPO = "brighter-dataset/BRIGHTER-emotion-intensities"
_REVISION = "08a5d61d3fc33036f97c8c76a61ff8d6f02f5157"

_EMO = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


class BrighterEmotionIntensityRegression(AbsTaskRegression):
    """Predict total emotional intensity (sum of 6 emotion intensities, 0-18)."""

    metadata = TaskMetadata(
        name="BrighterEmotionIntensityRegression",
        description=(
            "Predict the total emotional intensity of a native Brazilian-"
            "Portuguese social-media text (BRIGHTER, SemEval-2025 Task 11): the "
            "sum of the 0-3 intensity ratings across 6 emotions (anger, disgust, "
            "fear, joy, sadness, surprise), ranging 0-18."
        ),
        reference="https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities",
        dataset={
            "path": _REPO,
            "name": "ptbr",
            "revision": _REVISION,
        },
        type="Regression",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="kendalltau",
        date=("2024-01-01", "2025-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{brighter2025intensity,
    title        = {{BRIGHTER}: Bridging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages},
    author       = {{BRIGHTER consortium}},
    year         = {2025},
    howpublished = {SemEval-2025 Task 11 (Track B); HuggingFace dataset \texttt{brighter-dataset}},
    url          = {https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities},
}""",
    )

    label_column_name = "value"
    input_column_name = "text"

    def dataset_transform(self, **kwargs) -> None:
        """Total emotional intensity = sum of the 6 per-emotion 0-3 ratings."""
        self.dataset = self.dataset.map(
            lambda ex: {"value": float(sum(ex[e] for e in _EMO))}
        ).select_columns(["text", "value"])
