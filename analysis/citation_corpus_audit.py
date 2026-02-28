#!/usr/bin/env python3
"""Corpus-backed citation audit for the coherence-time paper.

This script is the "full" version of the embedding experiment:
- citation contexts come from the manuscript
- cited papers come from the paper-local bibliography snapshot
- full text is loaded from the shared literature store when available

Outputs:
- analysis/citation_corpus/corpus_audit_report.md
- analysis/citation_corpus/corpus_audit_results.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = PAPER_ROOT / "coherence_time.tex"
DATA_DIR = PAPER_ROOT / "analysis" / "citation_corpus"
BIB_PATH = DATA_DIR / "bibliography.json"
RESULTS_PATH = DATA_DIR / "corpus_audit_results.json"
REPORT_PATH = DATA_DIR / "corpus_audit_report.md"
EMBED_DIR = DATA_DIR / "embeddings"
CONTEXT_EMB_PATH = EMBED_DIR / "context_embeddings.npy"
DOC_EMB_PATH = EMBED_DIR / "document_embeddings.npy"
CHUNK_EMB_PATH = EMBED_DIR / "chunk_embeddings.npy"
PROBE_EMB_PATH = EMBED_DIR / "probe_embeddings.npy"
EMBED_INDEX_PATH = EMBED_DIR / "index.json"
SHARED_LITERATURE = Path("/Users/iantodd/Projects/highdimensional/literature")
SHARED_TEXT_DIR = SHARED_LITERATURE / "text"

PROBES = [
    (
        "Rare-event mixing support",
        "rare event first passage times in rapidly mixing stochastic processes "
        "with recurrence and approximately exponential hitting times",
    ),
    (
        "Ephaptic timing support",
        "extracellular electric fields modulate membrane potential and spike timing "
        "through ephaptic coupling",
    ),
    (
        "Psychedelic complexity support",
        "psychedelics increase neural signal complexity and alter conscious experience",
    ),
    (
        "Consciousness complexity index",
        "perturbational complexity index distinguishes conscious from unconscious states",
    ),
    (
        "Cerebellar fast motor control",
        "cerebellum supports rapid motor primitives with low coordination depth and fast timing",
    ),
    (
        "Entrainment timing prediction",
        "transcranial alternating current stimulation entrains brain oscillations and shifts temporal processing",
    ),
]


def extract_cite_keys(text: str) -> List[str]:
    keys: List[str] = []
    for match in re.finditer(r"\\(?:cite\w*)\{([^}]+)\}", text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key and key != "*":
                keys.append(key)
    return keys


def clean_latex(text: str) -> str:
    text = re.sub(r"\\(?:cite\w*|ref|eqref|label|url|href)\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " MATH ", text)
    text = re.sub(r"\\begin\{[^}]+\}", " ", text)
    text = re.sub(r"\\end\{[^}]+\}", " ", text)

    pattern = re.compile(r"\\[a-zA-Z*]+(?:\[[^\]]*\])?\{([^{}]*)\}")
    for _ in range(6):
        new_text = pattern.sub(r" \1 ", text)
        if new_text == text:
            break
        text = new_text

    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("~", " ")
    text = text.replace("---", " ")
    text = text.replace("--", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b10\.\S+", " ", text)
    text = re.sub(r"[{}_^%&]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_plaintext(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"-\s+\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentence_like_units(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", paragraph)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [paragraph.strip()]


def extract_citation_contexts(tex_text: str) -> List[Dict]:
    body = tex_text.split(r"\begin{thebibliography}", 1)[0]
    lines = body.splitlines()
    contexts: List[Dict] = []
    paragraph_lines: List[Tuple[int, str]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        start_line = paragraph_lines[0][0]
        paragraph = " ".join(line for _, line in paragraph_lines).strip()
        if "\\cite" in paragraph:
            for sentence in split_sentence_like_units(paragraph):
                cited_keys = extract_cite_keys(sentence)
                if not cited_keys:
                    continue
                cleaned = clean_latex(sentence)
                if len(cleaned.split()) < 4:
                    cleaned = clean_latex(paragraph)
                contexts.append(
                    {
                        "id": f"CTX{len(contexts) + 1:02d}",
                        "line": start_line,
                        "raw": sentence.strip(),
                        "text": cleaned,
                        "cited_keys": cited_keys,
                    }
                )
        paragraph_lines = []

    for lineno, line in enumerate(lines, start=1):
        if line.strip():
            paragraph_lines.append((lineno, line.rstrip()))
        else:
            flush_paragraph()
    flush_paragraph()
    return contexts


def load_bibliography(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = data.get("references", [])
    for ref in refs:
        raw_text = ref.get("raw_text") or ref.get("title") or ref.get("cite_key", "")
        ref["entry_text"] = clean_latex(raw_text)
    return refs


def chunk_text(text: str, target_words: int = 180, overlap_words: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= target_words:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    step = max(target_words - overlap_words, 1)
    while start < len(words):
        end = min(start + target_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def load_text_source_manifest(text_dir: Path) -> Dict[str, str]:
    manifest_path = text_dir.parent / "text_sources.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_corpus(refs: List[Dict], text_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    docs: List[Dict] = []
    chunks: List[Dict] = []
    source_manifest = load_text_source_manifest(text_dir)

    for ref in refs:
        cite_key = ref["cite_key"]
        text_path = text_dir / f"{cite_key}.txt"
        if text_path.exists():
            full_text = clean_plaintext(text_path.read_text(encoding="utf-8", errors="replace"))
            full_text = full_text[:80000]
            text_source = source_manifest.get(cite_key, "full_text")
            doc_text = full_text
        else:
            full_text = ""
            text_source = "title_only"
            doc_text = ref["entry_text"]

        doc = {
            "cite_key": cite_key,
            "title": ref.get("title", ""),
            "doi": ref.get("doi", ""),
            "year": ref.get("year", ""),
            "text_source": text_source,
            "doc_text": doc_text,
            "entry_text": ref["entry_text"],
        }
        docs.append(doc)

        source_chunks = chunk_text(doc_text)
        for idx, chunk in enumerate(source_chunks):
            chunks.append(
                {
                    "cite_key": cite_key,
                    "chunk_id": f"{cite_key}:{idx + 1}",
                    "text_source": text_source,
                    "text": chunk,
                }
            )

    return docs, chunks


def embed_groups(groups: Sequence[Tuple[str, Sequence[str]]]) -> Tuple[Dict[str, np.ndarray], Dict]:
    sizes = {name: len(texts) for name, texts in groups}
    all_texts = [text for _, texts in groups for text in texts]
    if not all_texts:
        return {name: np.zeros((0, 1)) for name, _ in groups}, {"backend": "empty"}

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        all_embeddings = np.asarray(model.encode(all_texts, show_progress_bar=False))
        backend = {"backend": "sentence-transformers", "model": "all-MiniLM-L6-v2"}
    except Exception as exc:  # pragma: no cover - environment-specific
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        all_embeddings = vectorizer.fit_transform(all_texts).toarray()
        backend = {
            "backend": "tfidf",
            "vocabulary_size": len(vectorizer.vocabulary_),
            "fallback_reason": str(exc),
        }

    outputs: Dict[str, np.ndarray] = {}
    start = 0
    for name, _ in groups:
        size = sizes[name]
        outputs[name] = all_embeddings[start:start + size]
        start += size
    return outputs, backend


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def nearest_items(row: np.ndarray, items: Sequence[Dict], k: int = 5) -> List[Dict]:
    order = np.argsort(row)[-k:][::-1]
    result = []
    for idx in order:
        result.append(
            {
                "cite_key": items[idx]["cite_key"],
                "similarity": float(row[idx]),
                "text_source": items[idx].get("text_source", ""),
            }
        )
    return result


def analyze_contexts(
    contexts: List[Dict],
    docs: List[Dict],
    doc_sims: np.ndarray,
    chunk_sims: np.ndarray,
    chunks: List[Dict],
) -> Dict:
    doc_lookup = {doc["cite_key"]: idx for idx, doc in enumerate(docs)}
    chunk_indices_by_key: Dict[str, List[int]] = {}
    for idx, chunk in enumerate(chunks):
        chunk_indices_by_key.setdefault(chunk["cite_key"], []).append(idx)

    strongest: List[Dict] = []
    review_candidates: List[Dict] = []
    coverage_gaps: List[Dict] = []
    top1_hits = 0
    top3_hits = 0
    full_text_backed = 0
    cited_scores: List[float] = []

    for i, context in enumerate(contexts):
        nearest_docs = nearest_items(doc_sims[i], docs, k=5)
        top_keys = [item["cite_key"] for item in nearest_docs]

        cited_records = []
        cited_full_text = False
        for key in context["cited_keys"]:
            doc_idx = doc_lookup.get(key)
            if doc_idx is None:
                continue
            doc = docs[doc_idx]
            doc_sim = float(doc_sims[i, doc_idx])
            best_chunk = None
            if key in chunk_indices_by_key:
                idxs = chunk_indices_by_key[key]
                best_idx = max(idxs, key=lambda idx: chunk_sims[i, idx])
                best_chunk = {
                    "similarity": float(chunk_sims[i, best_idx]),
                    "text": chunks[best_idx]["text"][:320],
                    "text_source": chunks[best_idx]["text_source"],
                }
            cited_records.append(
                {
                    "cite_key": key,
                    "doc_similarity": doc_sim,
                    "text_source": doc["text_source"],
                    "best_chunk": best_chunk,
                }
            )
            if doc["text_source"] == "full_text":
                cited_full_text = True

        cited_records.sort(key=lambda item: item["doc_similarity"], reverse=True)
        best_cited = cited_records[0] if cited_records else None
        best_cited_sim = best_cited["doc_similarity"] if best_cited else 0.0
        cited_scores.append(best_cited_sim)

        if nearest_docs and nearest_docs[0]["cite_key"] in context["cited_keys"]:
            top1_hits += 1
        if any(key in top_keys[:3] for key in context["cited_keys"]):
            top3_hits += 1
        if cited_full_text:
            full_text_backed += 1

        record = {
            **context,
            "top_matches": nearest_docs,
            "cited_matches": cited_records,
            "best_cited_similarity": best_cited_sim,
            "best_overall_similarity": nearest_docs[0]["similarity"] if nearest_docs else 0.0,
            "top_match_is_cited": bool(nearest_docs and nearest_docs[0]["cite_key"] in context["cited_keys"]),
            "any_cited_in_top3": any(key in top_keys[:3] for key in context["cited_keys"]),
            "has_full_text_cited_ref": cited_full_text,
        }

        if cited_records and best_cited:
            strongest.append(record)

        if not cited_full_text:
            coverage_gaps.append(record)
            continue

        if (
            not record["top_match_is_cited"]
            and not record["any_cited_in_top3"]
            and best_cited_sim < 0.35
        ):
            review_candidates.append(record)

    strongest.sort(key=lambda item: item["best_cited_similarity"], reverse=True)
    review_candidates.sort(key=lambda item: item["best_cited_similarity"])
    coverage_gaps.sort(key=lambda item: item["best_cited_similarity"])

    return {
        "summary": {
            "n_contexts": len(contexts),
            "n_docs": len(docs),
            "top1_hit_rate": top1_hits / max(len(contexts), 1),
            "top3_hit_rate": top3_hits / max(len(contexts), 1),
            "mean_best_cited_similarity": float(np.mean(cited_scores)) if cited_scores else 0.0,
            "median_best_cited_similarity": float(np.median(cited_scores)) if cited_scores else 0.0,
            "contexts_with_full_text_cited_ref": full_text_backed,
        },
        "strongest_contexts": strongest[:10],
        "review_candidates": review_candidates[:12],
        "coverage_gaps": coverage_gaps[:16],
    }


def orphan_references(docs: List[Dict], doc_sims: np.ndarray) -> List[Dict]:
    if doc_sims.size == 0:
        return []
    top3_by_context = np.argsort(doc_sims, axis=1)[:, -3:]
    covered_indices = set(int(idx) for row in top3_by_context for idx in row)
    orphans = []
    for idx, doc in enumerate(docs):
        if idx in covered_indices:
            continue
        max_sim = float(np.max(doc_sims[:, idx])) if doc_sims.shape[0] else 0.0
        orphans.append(
            {
                "cite_key": doc["cite_key"],
                "text_source": doc["text_source"],
                "max_similarity_to_any_context": max_sim,
                "title": doc["title"],
            }
        )
    orphans.sort(key=lambda item: item["max_similarity_to_any_context"])
    return orphans[:12]


def probe_corpus(probe_embs: np.ndarray, doc_embs: np.ndarray, docs: List[Dict]) -> List[Dict]:
    probe_sims = cosine_sim_matrix(probe_embs, doc_embs)
    output = []
    for (label, probe_text), row in zip(PROBES, probe_sims):
        matches = nearest_items(row, docs, k=3)
        output.append(
            {
                "label": label,
                "probe": probe_text,
                "matches": matches,
            }
        )
    return output


def build_report(results: Dict) -> str:
    summary = results["analysis"]["summary"]
    corpus = results["corpus_summary"]
    backend = results["backend"]
    lines: List[str] = []

    lines.append("# Citation Corpus Audit")
    lines.append("")
    lines.append(
        "This is a corpus-backed semantic audit. It compares citation contexts in "
        "`coherence_time.tex` against downloaded paper text when available, and "
        "falls back to auxiliary abstract/description text or bibliography-entry text "
        "when a paper has not been fully ingested yet."
    )
    lines.append("")
    lines.append("## Run Info")
    lines.append("")
    lines.append(f"- Backend: `{backend['backend']}`")
    if backend.get("model"):
        lines.append(f"- Model: `{backend['model']}`")
    if backend.get("fallback_reason"):
        lines.append(f"- Fallback reason: `{backend['fallback_reason']}`")
    lines.append(f"- Citation contexts analyzed: `{summary['n_contexts']}`")
    lines.append(f"- Bibliography entries analyzed: `{summary['n_docs']}`")
    lines.append(f"- References with DOI: `{corpus['refs_with_doi']}/{corpus['total_refs']}`")
    lines.append(
        f"- References with extracted full text: `{corpus['refs_with_full_text']}/{corpus['total_refs']}`"
    )
    lines.append(
        f"- References with abstract/description text: "
        f"`{corpus['refs_with_aux_text']}/{corpus['total_refs']}`"
    )
    lines.append(
        f"- Title-only fallback references: `{corpus['refs_title_only']}/{corpus['total_refs']}`"
    )
    lines.append(
        f"- Contexts with at least one full-text-backed cited reference: "
        f"`{summary['contexts_with_full_text_cited_ref']}/{summary['n_contexts']}`"
    )
    lines.append(f"- Top-1 cited match rate: `{summary['top1_hit_rate']:.2%}`")
    lines.append(f"- Top-3 cited match rate: `{summary['top3_hit_rate']:.2%}`")
    lines.append(
        f"- Mean best-cited similarity: `{summary['mean_best_cited_similarity']:.3f}`"
    )
    lines.append(
        f"- Median best-cited similarity: `{summary['median_best_cited_similarity']:.3f}`"
    )
    lines.append("")
    lines.append("## How To Read This")
    lines.append("")
    lines.append(
        "- Strong alignments are useful mainly as a sanity check that the citation graph is coherent."
    )
    lines.append(
        "- Review candidates matter more: these are contexts where a cited paper has full text in the corpus but still looks semantically weak relative to the sentence."
    )
    lines.append(
        "- Coverage gaps are not citation errors. They mean the cited paper does not yet have full-text support in the corpus, so the audit is still relying on auxiliary or metadata-level fallback."
    )
    lines.append("")
    lines.append("## Strongest Full-Text Alignments")
    lines.append("")
    for item in results["analysis"]["strongest_contexts"]:
        cited = next((m for m in item["cited_matches"] if m["text_source"] == "full_text"), None)
        if not cited:
            continue
        lines.append(
            f"- `coherence_time.tex:{item['line']}` `{item['id']}` best cited full-text match "
            f"`{cited['cite_key']}` at `{cited['doc_similarity']:.3f}`"
        )
        lines.append(f"  Context: {item['text']}")
        if cited["best_chunk"]:
            lines.append(f"  Best chunk: {cited['best_chunk']['text']}")
    if len(lines) and lines[-1] == "":
        pass
    lines.append("")
    lines.append("## Review Candidates")
    lines.append("")
    if not results["analysis"]["review_candidates"]:
        lines.append("No high-confidence full-text review candidates on this run.")
    for item in results["analysis"]["review_candidates"]:
        cited = item["cited_matches"][0] if item["cited_matches"] else None
        top = item["top_matches"][0] if item["top_matches"] else None
        lines.append(
            f"- `coherence_time.tex:{item['line']}` `{item['id']}` cited "
            f"`{', '.join(item['cited_keys'])}`"
        )
        lines.append(f"  Context: {item['text']}")
        if cited:
            lines.append(
                f"  Best cited full-text doc: `{cited['cite_key']}` at `{cited['doc_similarity']:.3f}`"
            )
            if cited["best_chunk"]:
                lines.append(f"  Best cited chunk: {cited['best_chunk']['text']}")
        if top:
            lines.append(
                f"  Best overall corpus match: `{top['cite_key']}` at `{top['similarity']:.3f}`"
            )
    lines.append("")
    lines.append("## Coverage Gaps")
    lines.append("")
    for item in results["analysis"]["coverage_gaps"]:
        lines.append(
            f"- `coherence_time.tex:{item['line']}` `{item['id']}` cited "
            f"`{', '.join(item['cited_keys'])}` has no full-text-backed cited reference yet"
        )
        lines.append(f"  Context: {item['text']}")
    lines.append("")
    lines.append("## Orphan References")
    lines.append("")
    for item in results["orphans"]:
        lines.append(
            f"- `{item['cite_key']}` `{item['text_source']}` max similarity "
            f"`{item['max_similarity_to_any_context']:.3f}`"
        )
        lines.append(f"  Title: {item['title']}")
    lines.append("")
    lines.append("## Probe Queries")
    lines.append("")
    for probe in results["probes"]:
        lines.append(f"- **{probe['label']}**")
        lines.append(f"  Probe: {probe['probe']}")
        for match in probe["matches"]:
            lines.append(
                f"  - `{match['cite_key']}` `{match['text_source']}` at `{match['similarity']:.3f}`"
            )
    lines.append("")
    lines.append("## Suggested Workflow")
    lines.append("")
    lines.append(
        "1. Ingest more cited papers into the shared literature store, or add auxiliary abstract/description text when full PDFs are unavailable."
    )
    lines.append(
        "2. Re-run this audit and focus on the review-candidate section, not the raw hit rate."
    )
    lines.append(
        "3. For any persistent mismatch, manually read the cited paper and either tighten the sentence or add a better source."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def save_embedding_artifacts(
    embedded: Dict[str, np.ndarray],
    backend: Dict,
    contexts: List[Dict],
    docs: List[Dict],
    chunks: List[Dict],
) -> None:
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CONTEXT_EMB_PATH, embedded["contexts"])
    np.save(DOC_EMB_PATH, embedded["docs"])
    np.save(CHUNK_EMB_PATH, embedded["chunks"])
    np.save(PROBE_EMB_PATH, embedded["probes"])

    index = {
        "backend": backend,
        "paths": {
            "context_embeddings": str(CONTEXT_EMB_PATH),
            "document_embeddings": str(DOC_EMB_PATH),
            "chunk_embeddings": str(CHUNK_EMB_PATH),
            "probe_embeddings": str(PROBE_EMB_PATH),
        },
        "counts": {
            "contexts": len(contexts),
            "documents": len(docs),
            "chunks": len(chunks),
            "probes": len(PROBES),
        },
        "contexts": [
            {
                "id": item["id"],
                "line": item["line"],
                "cited_keys": item["cited_keys"],
                "text": item["text"],
            }
            for item in contexts
        ],
        "documents": [
            {
                "cite_key": item["cite_key"],
                "title": item["title"],
                "year": item["year"],
                "doi": item["doi"],
                "text_source": item["text_source"],
            }
            for item in docs
        ],
        "chunks": [
            {
                "chunk_id": item["chunk_id"],
                "cite_key": item["cite_key"],
                "text_source": item["text_source"],
                "preview": item["text"][:240],
            }
            for item in chunks
        ],
        "probes": [{"label": label, "text": text} for label, text in PROBES],
    }
    EMBED_INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    shared_text_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SHARED_TEXT_DIR
    tex_text = TEX_PATH.read_text(encoding="utf-8", errors="replace")
    contexts = extract_citation_contexts(tex_text)
    refs = load_bibliography(BIB_PATH)
    docs, chunks = build_corpus(refs, shared_text_dir)

    groups = [
        ("contexts", [ctx["text"] for ctx in contexts]),
        ("docs", [doc["doc_text"] for doc in docs]),
        ("chunks", [chunk["text"] for chunk in chunks]),
        ("probes", [probe for _, probe in PROBES]),
    ]
    embedded, backend = embed_groups(groups)

    doc_sims = cosine_sim_matrix(embedded["contexts"], embedded["docs"])
    chunk_sims = cosine_sim_matrix(embedded["contexts"], embedded["chunks"])
    analysis = analyze_contexts(contexts, docs, doc_sims, chunk_sims, chunks)
    orphans = orphan_references(docs, doc_sims)
    probes = probe_corpus(embedded["probes"], embedded["docs"], docs)
    save_embedding_artifacts(embedded, backend, contexts, docs, chunks)

    refs_with_doi = sum(1 for ref in refs if ref.get("doi"))
    refs_with_full_text = sum(1 for doc in docs if doc["text_source"] == "full_text")
    refs_with_aux_text = sum(
        1 for doc in docs if doc["text_source"] not in {"full_text", "title_only"}
    )
    payload = {
        "backend": backend,
        "corpus_summary": {
            "total_refs": len(refs),
            "refs_with_doi": refs_with_doi,
            "refs_with_full_text": refs_with_full_text,
            "refs_with_aux_text": refs_with_aux_text,
            "refs_title_only": len(refs) - refs_with_full_text - refs_with_aux_text,
            "shared_text_dir": str(shared_text_dir),
        },
        "analysis": analysis,
        "orphans": orphans,
        "probes": probes,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(build_report(payload), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
