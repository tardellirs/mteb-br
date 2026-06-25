"""EnemEssayRegression — predict the ENEM essay grade from the essay text.

AES-ENEM corpus (kamel-usp/aes_enem_dataset, PROPOR 2024 config): native
Brazilian high-school argumentative essays graded on the 5 official ENEM
competencies (0-200 each); the regression target is the TOTAL grade
(0-1000 = sum of the five). First MTEB(por) task of the Regression task-type.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks.regression import AbsTaskRegression

_REPO = "kamel-usp/aes_enem_dataset"
_REVISION = "8df4b700221dc20410e1e28900bbf215da545347"


class EnemEssayRegression(AbsTaskRegression):
    """Predict the total ENEM essay grade (0-1000) from the essay text."""

    metadata = TaskMetadata(
        name="EnemEssayRegression",
        description=(
            "Predict the total grade (0-1000) of a Brazilian high-school ENEM "
            "argumentative essay from its text. The total is the sum of the five "
            "official ENEM competency scores (0-200 each). Native PT-BR essays "
            "from the AES-ENEM corpus (ENEM 2016-2020 prompts)."
        ),
        reference="https://huggingface.co/datasets/kamel-usp/aes_enem_dataset",
        dataset={
            "path": _REPO,
            "name": "PROPOR2024",
            "revision": _REVISION,
        },
        type="Regression",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="kendalltau",
        date=("2016-01-01", "2020-12-31"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{aes_enem_dataset,
    title        = {{AES-ENEM}: an Automated Essay Scoring dataset of {B}razilian {ENEM} argumentative essays},
    author       = {{kamel-usp}},
    howpublished = {HuggingFace dataset \texttt{kamel-usp/aes\_enem\_dataset}, presented at PROPOR 2024},
    url          = {https://huggingface.co/datasets/kamel-usp/aes_enem_dataset},
}""",
    )

    label_column_name = "value"
    input_column_name = "text"

    def dataset_transform(self, **kwargs) -> None:
        """essay_text -> text; total grade (grades[5], 0-1000) -> value (continuous)."""
        self.dataset = self.dataset.map(
            lambda ex: {"text": ex["essay_text"], "value": float(ex["grades"][5])}
        ).select_columns(["text", "value"])
