<div align="center">

# MTEB-PT

**Massive Text Embedding Benchmark for Brazilian Portuguese**

[![Leaderboard](https://img.shields.io/badge/🤗-Leaderboard-yellow)](https://huggingface.co/spaces/mteb-pt/leaderboard)
[![Results](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/mteb-pt/mteb-pt-results)
[![Org](https://img.shields.io/badge/🤗-mteb--pt-green)](https://huggingface.co/mteb-pt)
[![License: Apache 2.0](https://img.shields.io/badge/License_(code)-Apache_2.0-blue)](LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/License_(data)-CC--BY--4.0-orange)](https://creativecommons.org/licenses/by/4.0/)

</div>

A public benchmark for evaluating text embedding models on **native Brazilian Portuguese**, built as a thin extension on top of the [`mteb`](https://github.com/embeddings-benchmark/mteb) library.

**Live leaderboard**: <https://huggingface.co/spaces/mteb-pt/leaderboard>

## What it is

- **26 tasks** from **native** PT-BR sources — created or mined in Portuguese, **no machine translation**
- **8 MTEB task-types**: Classification, Multi-label Classification, Regression, Pair Classification, STS, Clustering, Retrieval, Reranking
- Open-weights **and** commercial-API models evaluated
- Per-task scores, per-query parquets, and reproduction scripts all public

> **v2** (this release) expands the original 16-task suite to 26 native tasks, adding two task-types (Regression, Multi-label Classification) and new domains (Emotion, Financial, News/fact-check, Government/legislative, Programming). See [`HEADLINE_TASKS`](mteb_pt/__init__.py).

## Quickstart

Install the package directly from GitHub:

```bash
pip install git+https://github.com/tardellirs/mteb-pt.git
```

Evaluate any model on a single MTEB-PT task:

```python
import mteb_pt.register   # registers the 26 tasks with the global mteb registry
import mteb

model = mteb.get_model("intfloat/multilingual-e5-large-instruct")
task  = mteb.get_task("HateBR")
mteb.evaluate(model, tasks=[task])
```

Or run the **full 26-task suite** in one command (resumable; spot / block-volume friendly):

```bash
python scripts/run_mteb_por_v2.py intfloat/multilingual-e5-large-instruct
```

Re-running the same command **resumes**: finished `(model, task)` pairs are skipped (`overwrite_strategy="only-missing"`). Point `HF_HOME` and `MTEB_CACHE` at a persistent volume to survive spot preemption without re-downloading. See the script header for the full setup.

Compute the **paired-bootstrap p-value** between two model evaluations:

```bash
python examples/compute_bootstrap_ci.py \
    --results-a ./results/intfloat__multilingual-e5-large-instruct \
    --results-b ./results/Qwen__Qwen3-Embedding-8B
```

## Package layout

```
mteb_pt/
├── __init__.py                       # HEADLINE_TASKS (26) + TASKS_BY_CATEGORY map
├── register.py                       # side-effect: registers the 26 tasks with mteb
├── stats.py                          # bootstrap CIs + paired significance helpers
└── tasks/
    ├── classification/por/           # HateBR, ToxSynPT, FactckBr
    ├── multilabel_classification/por/ # BrighterEmotion, OlidBr
    ├── regression/por/               # EnemEssay, NarrativeEssaysBR, BrighterEmotionIntensity
    ├── pair_classification/por/      # AssinRTE, InferBR
    ├── sts/por/                      # AssinSTS  (+ Assin2STS upstream)
    ├── clustering/por/               # MedPT, WikipediaPTCategories, JurisTCU-P2P,
    │                                 #   SciELO, CamaraProposicoes, StackoverflowPt
    ├── retrieval/por/                # Quati, JurisTCU, BRTaxQAR, FaQuADIR,
    │                                 #   MedPTRetrieval, FaqBacen
    └── reranking/por/                # QuatiReranking, JurisTCUReranking

scripts/
└── run_mteb_por_v2.py                # full 26-task suite, resumable (spot/block-volume aware)

examples/
├── quickstart.py                     # 1 model × 1 task smoke test
└── compute_bootstrap_ci.py           # paired-bootstrap p-value between two models

tests/
└── test_register.py                  # smoke tests: all 26 tasks resolve via mteb.get_task
```

## Task suite (26 tasks)

Each task wrapper pins its source dataset to a specific revision SHA. All sources are native PT-BR (no machine translation).

| Task | Type | Source |
|---|---|---|
| [HateBR](https://aclanthology.org/2022.lrec-1.777/) | Classification | Vargas et al. 2022 — [HF](https://huggingface.co/datasets/franciellevargas/HateBR) |
| [ToxSynPT](https://huggingface.co/datasets/AKCIT/ToxSyn-PT) | Classification | AKCIT |
| [FactckBrClassification](https://huggingface.co/datasets/tardellirs/mteb-pt-factckbr) | Classification | FACTCK.BR fact-check claims |
| [BrighterEmotionMultilabelClassification](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories) | Multi-label Classification | BRIGHTER (multi-emotion) |
| [OlidBrMultilabelClassification](https://huggingface.co/datasets/dougtrajano/olid-br) | Multi-label Classification | OLID-BR (offensive types) |
| [EnemEssayRegression](https://huggingface.co/datasets/kamel-usp/aes_enem_dataset) | Regression | AES-ENEM essay grades |
| [NarrativeEssaysBRRegression](https://huggingface.co/datasets/tardellirs/mteb-pt-narrative-essays) | Regression | Narrative essay competencies |
| [BrighterEmotionIntensityRegression](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities) | Regression | BRIGHTER emotion intensity |
| [AssinRTE](https://aclanthology.org/2020.lrec-1.319/) | Pair Classification (NLI) | Real et al. 2020 — [HF](https://huggingface.co/datasets/nilc-nlp/assin) |
| [InferBR](https://aclanthology.org/2024.lrec-main.788/) | Pair Classification (NLI) | Rodrigues et al. 2024 |
| [AssinSTS](https://aclanthology.org/2020.lrec-1.319/) | STS | Real et al. 2020 — [HF](https://huggingface.co/datasets/nilc-nlp/assin) |
| [Assin2STS](https://huggingface.co/datasets/nilc-nlp/assin2) | STS | ASSIN 2 (NILC) — *upstream mteb* |
| [WikipediaPTCategoriesClusteringP2P](https://huggingface.co/datasets/tardellirs/mteb-pt-wikipedia-categories) | Clustering | Wikipedia-derived (this benchmark) |
| [MedPTClustering](https://huggingface.co/datasets/AKCIT/MedPT) | Clustering | AKCIT |
| [JurisTCUClusteringP2P](https://huggingface.co/datasets/LeandroRibeiro/JurisTCU) | Clustering | TCU rulings (this benchmark) |
| [SciELOClusteringP2P](https://huggingface.co/datasets/tardellirs/mteb-pt-scielo-clustering) | Clustering | SciELO abstracts (this benchmark) |
| [CamaraProposicoesClustering](https://huggingface.co/datasets/tardellirs/mteb-pt-camara-proposicoes-clustering) | Clustering | Câmara dos Deputados (LAI / CC-BY) |
| [StackoverflowPtClustering](https://huggingface.co/datasets/tardellirs/mteb-pt-stackoverflow-clustering) | Clustering | Stack Overflow em Português (CC-BY-SA) |
| [Quati](https://aclanthology.org/2024.stil-1.19/) | Retrieval | Bueno et al. 2024 — [HF](https://huggingface.co/datasets/unicamp-dl/quati) (250k subsample) |
| [JurisTCU](https://huggingface.co/datasets/LeandroRibeiro/JurisTCU) | Retrieval | Ribeiro et al. — TCU rulings |
| [BRTaxQAR](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R) | Retrieval | UNICAMP-DL |
| [FaQuADIR](https://github.com/liafacom/faquad) | Retrieval | Sayama et al. 2019 |
| [MedPTRetrieval](https://huggingface.co/datasets/AKCIT/MedPT) | Retrieval | AKCIT |
| [FaqBacenRetrieval](https://huggingface.co/datasets/tardellirs/mteb-pt-faq-bacen) | Retrieval | Banco Central do Brasil FAQ |
| [QuatiReranking](https://aclanthology.org/2024.stil-1.19/) | Reranking | Bueno et al. 2024 — BM25 hard negatives |
| [JurisTCUReranking](https://huggingface.co/datasets/LeandroRibeiro/JurisTCU) | Reranking | TCU rulings — BM25 hard negatives |

If you cite a specific task, please cite its **original source** alongside this benchmark.

## Submit a new model

Two channels, pick whichever fits:

- **HF Discussion** on the leaderboard Space → [open a thread](https://huggingface.co/spaces/mteb-pt/leaderboard/discussions/new) and attach the eval JSONs
- **GitHub Issue** → use the [model submission template](https://github.com/tardellirs/mteb-pt/issues/new?template=submit-model.yml)

Required: (1) `model_id`; (2) per-task result JSONs for the 26 tasks; (3) a reproducible evaluation command (e.g. `python scripts/run_mteb_por_v2.py <model_id>`). We re-run a sample of submissions before merging. Closed-API models accepted (verified against the vendor's official endpoint).

## Propose a new task

A task is a candidate for inclusion if it:
- Sources its data from **native** PT-BR (not machine-translated)
- Has clear, permissive licensing
- Discriminates across embedding models (i.e. not degenerate)

Open an [issue using the task proposal template](https://github.com/tardellirs/mteb-pt/issues/new?template=propose-task.yml) describing the dataset, license, size, and discrimination evidence.

## Maintainer

**Tardelli Stekel** — IFSP, São Paulo, Brazil
Email: <stekel@ifsp.edu.br>

Contributions, corrections, and discussion all welcome via Issues or HF Discussions.

## Citation

```bibtex
@misc{mteb-portuguese-2026,
  title  = {MTEB Portuguese: A Massive Text Embedding Benchmark for Brazilian Portuguese},
  author = {Stekel, Tardelli},
  year   = {2026},
  url    = {https://huggingface.co/spaces/mteb-pt/leaderboard}
}
```

If you used a specific task novel to this benchmark, please also cite the original task dataset.

## License

- Benchmark code: Apache-2.0
- Results dataset: CC-BY-4.0
- Individual task datasets: see each dataset's original license (linked in the task table above)
- Models evaluated: see each model card

## Acknowledgments

Built on top of the [`mteb`](https://github.com/embeddings-benchmark/mteb) library (Muennighoff et al., 2023). The multilingual sub-benchmark methodology follows MMTEB (Enevoldsen et al., 2025). Task datasets contributed by their original authors — see the [Task suite](#task-suite-26-tasks) table for sources and citations.
