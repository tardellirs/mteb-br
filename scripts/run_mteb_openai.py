#!/usr/bin/env python
"""Run MTEB(por, v2) for an OpenAI embedding model via the OpenAI API (DIRECT),
using the Batch API (50% off) for big corpora.

Hybrid: big encode calls (corpora > OPENAI_BATCH_THRESHOLD texts) -> OpenAI Batch API
($0.065/1M for 3-large, 50% off; file-based JSONL, concurrent batches); small encode
calls -> sync embeddings (input list up to 2048). mteb AbsEncoder so scores match the
open-weight + gateway fleets. HF-resume (pull at start, daemon sync, only-missing).

Env: OPENAI_API_KEY, OPENAI_MODEL (default text-embedding-3-large), HF_RESULTS_REPO,
     MTEB_CACHE, HF_SYNC_SECONDS, OPENAI_BATCH_THRESHOLD (default 2000), MTEB_TASKS.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import threading
import time

import numpy as np

import mteb
import mteb_pt
import mteb_pt.register as register
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.abs_encoder import AbsEncoder
from huggingface_hub import HfApi, snapshot_download
from openai import OpenAI

REPO = os.environ.get("HF_RESULTS_REPO", "mteb-pt/mteb-pt-results")
CACHE = os.environ.get("MTEB_CACHE", os.path.expanduser("~/.cache/mteb"))
RESULTS = os.path.join(CACHE, "results")
SYNC_EVERY = int(os.environ.get("HF_SYNC_SECONDS", "120"))
TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("OPENAI_MODEL", "text-embedding-3-large")
DIM = 3072 if "large" in MODEL_ID else 1536
BATCH_THRESHOLD = int(os.environ.get("OPENAI_BATCH_THRESHOLD", "2000"))
CHUNK_REQS = 45000  # < 50k requests/batch
_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}


def v2_tasks() -> list:
    tasks = [cls() for cls in register._TASKS_TO_REGISTER if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    only = os.environ.get("MTEB_TASKS")
    if only:
        keep = {x.strip() for x in only.split(",")}
        tasks = [t for t in tasks if t.metadata.name in keep]
    tasks.sort(key=lambda t: _PRIORITY.get(t.metadata.type, 9))
    return tasks


class OpenAIBatchModel(AbsEncoder):
    """mteb encoder for an OpenAI embedding model (sync small, Batch API for big)."""

    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # direct OpenAI
        self.mteb_model_meta = ModelMeta(
            loader=None, name=f"openai/{MODEL_ID}", revision="api",
            release_date="2024-01-25", languages=["por-Latn"], n_parameters=None,
            memory_usage_mb=None, max_tokens=8191, embed_dim=DIM, license=None,
            open_weights=False, public_training_code=None, public_training_data=None,
            framework=["API"], similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False, training_datasets=None,
        )

    def _embed_sync(self, texts):
        out = []
        for i in range(0, len(texts), 2048):
            chunk = [(t[:30000] if t else " ") or " " for t in texts[i:i + 2048]]
            for delay in (2, 5, 15, 30, 60, None):
                try:
                    r = self.client.embeddings.create(model=MODEL_ID, input=chunk)
                    out.extend([d.embedding for d in r.data]); break
                except Exception as e:  # noqa: BLE001
                    if delay is None:
                        print(f"  [openai] sync give-up: {str(e)[:90]}", flush=True)
                        out.extend([[0.0] * DIM] * len(chunk)); break
                    time.sleep(delay)
        return np.array(out, dtype=np.float32)

    def _submit_chunk(self, texts, start):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for j, t in enumerate(texts):
            f.write(json.dumps({
                "custom_id": str(start + j), "method": "POST", "url": "/v1/embeddings",
                "body": {"model": MODEL_ID, "input": (t[:30000] if t else " ") or " "},
            }) + "\n")
        f.close()
        for delay in (5, 15, 30, 60, None):
            try:
                up = self.client.files.create(file=open(f.name, "rb"), purpose="batch")
                b = self.client.batches.create(input_file_id=up.id, endpoint="/v1/embeddings",
                                                completion_window="24h")
                os.unlink(f.name)
                return b.id
            except Exception as e:  # noqa: BLE001
                if delay is None:
                    os.unlink(f.name); raise
                print(f"  [openai] submit retry ({str(e)[:70]})", flush=True); time.sleep(delay)

    def _embed_batch(self, texts):
        """OpenAI Batch API (50% off): submit all chunks concurrently, poll, assemble."""
        batch_ids = []
        for ci in range(0, len(texts), CHUNK_REQS):
            batch_ids.append(self._submit_chunk(texts[ci:ci + CHUNK_REQS], ci))
        print(f"  [openai] Batch: {len(texts)} texts -> {len(batch_ids)} batch(es) submitted, polling", flush=True)
        results = {}
        pending = set(batch_ids)
        while pending:
            time.sleep(45)
            for bid in list(pending):
                try:
                    b = self.client.batches.retrieve(bid)
                except Exception:  # noqa: BLE001
                    continue
                if b.status in ("completed", "failed", "expired", "cancelled"):
                    pending.discard(bid)
                    if b.status == "completed" and b.output_file_id:
                        content = self.client.files.content(b.output_file_id).read()
                        for line in content.decode().splitlines():
                            o = json.loads(line); resp = o.get("response", {})
                            if resp.get("status_code") == 200:
                                data = resp.get("body", {}).get("data", [])
                                if data:
                                    results[o["custom_id"]] = np.array(data[0]["embedding"], dtype=np.float32)
                    else:
                        print(f"  [openai] batch {bid} {b.status} (some texts -> zero vec)", flush=True)
        return np.array([results.get(str(i), np.zeros(DIM, dtype=np.float32)) for i in range(len(texts))], dtype=np.float32)

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        texts = [text for batch in inputs for text in batch["text"]]
        if len(texts) > BATCH_THRESHOLD:
            return self._embed_batch(texts)
        return self._embed_sync(texts)


def _n_files(root):
    return sum(len(f) for _, _, f in os.walk(root)) if os.path.isdir(root) else 0


def pull_from_hf():
    os.makedirs(RESULTS, exist_ok=True)
    try:
        snapshot_download(REPO, repo_type="dataset", allow_patterns="results/**", local_dir=CACHE, token=TOKEN)
        print(f"[hf-resume] pulled {_n_files(RESULTS)} files", flush=True)
    except Exception as e:
        print(f"[hf-resume] pull skipped ({str(e)[:90]})", flush=True)


def _upload_once(api):
    if _n_files(RESULTS) == 0:
        return
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=RESULTS, path_in_repo="results", repo_id=REPO,
                              repo_type="dataset", commit_message="openai sync")
            return
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1) + random.uniform(0, 5)); continue
            print(f"[hf-resume] sync err ({str(e)[:90]})", flush=True); return


def sync_loop(stop, api):
    time.sleep(SYNC_EVERY * random.uniform(0.5, 1.5))
    while not stop.is_set():
        _upload_once(api); stop.wait(SYNC_EVERY)


def main():
    pull_from_hf()
    tasks = v2_tasks()
    print(f"[openai] {MODEL_ID} x {len(tasks)} tasks | batch>{BATCH_THRESHOLD} texts | dim={DIM}", flush=True)
    api = HfApi(token=TOKEN)
    stop = threading.Event()
    threading.Thread(target=sync_loop, args=(stop, api), daemon=True).start()
    t0 = time.time()
    print(f"\n=== model: openai/{MODEL_ID} ===", flush=True)
    try:
        mteb.evaluate(OpenAIBatchModel(), tasks=tasks, overwrite_strategy="only-missing",
                      encode_kwargs={"batch_size": 256}, raise_error=False)
        print(f"=== openai done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"=== openai FAILED: {type(e).__name__}: {str(e)[:200]} ===", flush=True)
    stop.set()
    _upload_once(api)
    print("=== FLEET RUN COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
