"""NarrativeEssaysBRRegression — grade a 5th-grade Brazilian narrative essay.

NarrativeEssaysBR (moesiof/portuguese-narrative-essays, Kaggle, CC-BY-4.0):
native PT-BR elementary-school (5th-grade) narrative essays graded on 4
competencies (formal register, thematic coherence, narrative/rhetorical
structure, cohesion; 1-5 each). The regression target is the TOTAL (sum, 4-20).
Complements the argumentative-essay AES task (EnemEssayRegression) with a
distinct genre (narrative) and age group (elementary). Mirror pinned at
tardellirs/mteb-pt-narrative-essays.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks.regression import AbsTaskRegression

_REPO = "tardellirs/mteb-pt-narrative-essays"
_REVISION = "f83c5198ef498d34be6ef2a5b5cf80df3d690966"


class NarrativeEssaysBRRegression(AbsTaskRegression):
    """Predict the total competency score (4-20) of a 5th-grade narrative essay."""

    metadata = TaskMetadata(
        name="NarrativeEssaysBRRegression",
        description=(
            "Predict the total grade (sum of 4 competency scores, 4-20) of a "
            "native Brazilian-Portuguese 5th-grade narrative essay. Competencies: "
            "formal register, thematic coherence, narrative/rhetorical structure, "
            "cohesion. Distinct genre (narrative) and age group (elementary) from "
            "the ENEM argumentative-essay AES task."
        ),
        reference="https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Regression",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="kendalltau",
        date=("2023-01-01", "2024-12-31"),
        domains=["Academic", "Fiction", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{narrative_essays_br,
    title        = {Portuguese Narrative Essays: a {B}razilian elementary-school narrative-essay corpus with competency scores},
    author       = {{moesiof}},
    year         = {2024},
    doi          = {10.34740/kaggle/ds/4464018},
    howpublished = {Kaggle dataset},
    url          = {https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays},
}""",
    )

    label_column_name = "value"
    input_column_name = "text"
