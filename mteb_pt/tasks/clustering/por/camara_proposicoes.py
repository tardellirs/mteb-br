"""CamaraProposicoesClustering — cluster Brazilian Chamber bills by legislative theme.

Ementas (official summaries) of bills (proposições) from the Brazilian Chamber
of Deputies open-data API (dadosabertos.camara.leg.br), labelled by the Chamber's
official theme taxonomy (24 themes with enough volume kept: Economia, Educação,
Saúde, Meio Ambiente e Desenvolvimento Sustentável, Direitos Humanos e Minorias,
etc.). Public-domain government data (Lei de Acesso à Informação 12.527/2011).
Government / legislative domain. Built as tardellirs/mteb-pt-camara-proposicoes-clustering.
"""

from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

_REPO = "tardellirs/mteb-pt-camara-proposicoes-clustering"
_REVISION = "1eafc7583e0d23e121f182c0bc067145a2d5ef73"


class CamaraProposicoesClustering(AbsTaskClustering):
    """Cluster Brazilian Chamber bill summaries (ementas) into legislative themes."""

    metadata = TaskMetadata(
        name="CamaraProposicoesClustering",
        description=(
            "Cluster the summaries (ementas) of bills from the Brazilian Chamber "
            "of Deputies into legislative themes from the Chamber's official "
            "taxonomy (Economia, Educação, Saúde, Meio Ambiente, Direitos "
            "Humanos, Administração Pública, etc.). Native PT-BR legislative "
            "text; public-domain government open data."
        ),
        reference="https://dadosabertos.camara.leg.br/",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2019-01-01", "2024-12-31"),
        domains=["Government", "Legal", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""@misc{camara_dados_abertos,
    title        = {Dados Abertos da C{\^a}mara dos Deputados — Proposi{\c{c}}{\~o}es},
    author       = {{C{\^a}mara dos Deputados}},
    howpublished = {Open-data API, \url{https://dadosabertos.camara.leg.br/}, under the Brazilian Access to Information Law (Lei 12.527/2011)},
}""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
