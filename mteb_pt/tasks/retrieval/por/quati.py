"""Quati — native Brazilian Portuguese web retrieval (Bueno et al., STIL 2024).

Reference: https://aclanthology.org/2024.stil-1.19/
Dataset:   https://huggingface.co/datasets/unicamp-dl/quati

Quati ships 50 native PT-BR test topics + ~1,900 human relevance judgements over
a 1M Brazilian web passage pool (ClueWeb22-PT) — the first retrieval benchmark
designed natively for PT-BR (no translation).

For tractability we evaluate over a **250k subsample** of the corpus that keeps
ALL judged passages plus a fixed random sample (seed 42) of the rest — an
accepted MTEB practice for very large corpora. This keeps Quati a hard
large-scale retrieval task while making the suite ~4x cheaper to run and re-run
(Quati was ~93% of the per-model encoding load at 1M). Repackaged + pinned at
tardellirs/mteb-pt-quati-250k (corpus / queries / qrels configs).
"""

from __future__ import annotations

from typing import Any

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval

_REPO = "tardellirs/mteb-pt-quati-250k"
_REVISION = "7440d16aa3a53c037e63a16591c461210b72dd82"


class Quati(AbsTaskRetrieval):
    """Quati — Brazilian Portuguese native web retrieval, 50 topics over a 250k pool."""

    metadata = TaskMetadata(
        name="Quati",
        description=(
            "Quati: native Brazilian Portuguese web retrieval benchmark. 50 test "
            "topics with ~1,900 human relevance judgements over a Brazilian web "
            "passage pool (ClueWeb22-PT). First PT-BR retrieval benchmark designed "
            "without translation; queries are naturally-occurring Brazilian web "
            "questions. Evaluated over a 250k corpus subsample (all judged passages "
            "plus a fixed random sample of the 1M pool) for tractability."
        ),
        reference="https://aclanthology.org/2024.stil-1.19/",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2024-06-30"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""@inproceedings{bueno-etal-2024-quati,
    title = "{Q}uati: A {B}razilian {P}ortuguese Information Retrieval Dataset from Native Speakers",
    author = "Bueno, Mirelle  and
      Maia Sanches, Eduardo  and
      Lotufo, Roberto  and
      Nogueira, Rodrigo",
    booktitle = "Proceedings of the 16th Symposium in Information and Human Language Technology (STIL 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.stil-1.19/",
}""",
    )

    def load_data(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Load corpus/queries/qrels from the pinned 250k-subsample HF dataset."""
        if getattr(self, "data_loaded", False):
            return
        from datasets import load_dataset

        corpus = {
            r["_id"]: {"text": r["text"], "title": r["title"]}
            for r in load_dataset(_REPO, "corpus", split="test", revision=_REVISION)
        }
        queries = {
            r["_id"]: r["text"]
            for r in load_dataset(_REPO, "queries", split="test", revision=_REVISION)
        }
        relevant_docs: dict[str, dict[str, int]] = {}
        for r in load_dataset(_REPO, "qrels", split="test", revision=_REVISION):
            relevant_docs.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
