"""BrighterEmotionMultilabelClassification — multi-label emotion recognition (PT-BR).

BRIGHTER dataset (SemEval-2025 Task 11) ptbr split: native PT-BR social-media
texts annotated by native speakers for the presence of 6 emotions (anger,
disgust, fear, joy, sadness, surprise). Multi-label — a text may carry several
emotions or none. First MTEB(por) MultilabelClassification task.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification

_REPO = "brighter-dataset/BRIGHTER-emotion-categories"
_REVISION = "419566ac8f46f951b51584be36c13e5363008144"


class BrighterEmotionMultilabelClassification(AbsTaskMultilabelClassification):
    """Multi-label emotion presence (6 emotions) on native PT-BR social-media texts."""

    metadata = TaskMetadata(
        name="BrighterEmotionMultilabelClassification",
        description=(
            "Multi-label emotion recognition on native Brazilian-Portuguese "
            "social-media texts (BRIGHTER, SemEval-2025 Task 11). Each text is "
            "labelled with the subset of 6 emotions present (anger, disgust, "
            "fear, joy, sadness, surprise); a text may have several or none."
        ),
        reference="https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories",
        dataset={
            "path": _REPO,
            "name": "ptbr",
            "revision": _REVISION,
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2025-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{brighter2025,
    title        = {{BRIGHTER}: Bridging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages},
    author       = {{BRIGHTER consortium}},
    year         = {2025},
    howpublished = {SemEval-2025 Task 11; HuggingFace dataset \texttt{brighter-dataset}},
    url          = {https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories},
}""",
    )

    input_column_name = "text"
    label_column_name = "label"

    def dataset_transform(self, **kwargs) -> None:
        """The `emotions` list column is the multi-label target."""
        self.dataset = self.dataset.rename_columns({"emotions": "label"}).select_columns(
            ["text", "label"]
        )
