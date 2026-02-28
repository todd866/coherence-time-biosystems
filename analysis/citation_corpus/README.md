# Citation Corpus Audit

This folder holds paper-specific outputs for a corpus-backed citation audit.
The infrastructure stays in `/Users/iantodd/Projects/highdimensional/tools/research-engine`
and the downloaded PDF/text corpus lives outside the paper repo.

## What this is for

The goal is not just to compare manuscript sentences against bibliography titles.
The stronger version is:

1. Extract the paper's references.
2. Resolve DOIs.
3. Verify those DOIs against CrossRef metadata.
4. Download open-access PDFs into a shared corpus store.
5. Extract text from those PDFs.
6. Embed citation contexts from `coherence_time.tex` and compare them against the
   downloaded paper text.

When a cited paper cannot be sourced as a full PDF, the corpus can also carry
paper-specific auxiliary text (abstracts, structured descriptions, or local self-citation
full text) so the embedding audit is not forced to fall back all the way to titles.

That lets us flag:

- claims whose cited paper does not look semantically close even at full-text level
- references that are present in the bibliography but semantically orphaned
- missing corpus coverage, which tells us where the audit is still weak

## Current files

- `bibliography.json`: extracted paper-local bibliography snapshot
- `missing_dois.json`: references still lacking DOI after resolution
- `doi_resolution_log.json`: resolved DOI assignments
- `verification_report.json`: CrossRef verification results
- `corpus_audit_report.md`: current corpus-backed audit report
- `corpus_audit_results.json`: machine-readable audit output
- `manual_citation_review.md`: manual triage of the most suspicious embedding hits

## Shared corpus location

The shared cache for this paper lives at:

`/Users/iantodd/Projects/highdimensional/literature/coherence_time_corpus`

Expected structure:

- `bibliography.json`
- `pdfs/*.pdf`
- `text/*.txt`
- `text_sources.json` (optional manifest for non-full-text entries such as abstract or description text)

## Rerun commands

```bash
cd /Users/iantodd/Projects/highdimensional/tools/research-engine

python3 -m research_engine extract \
  /Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time \
  -o /Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/analysis/citation_corpus/bibliography.json

python3 -m research_engine resolve \
  /Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/analysis/citation_corpus/bibliography.json

python3 -m research_engine verify \
  /Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/analysis/citation_corpus/bibliography.json

mkdir -p /Users/iantodd/Projects/highdimensional/literature/coherence_time_corpus
cp /Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/analysis/citation_corpus/bibliography.json \
  /Users/iantodd/Projects/highdimensional/literature/coherence_time_corpus/bibliography.json

python3 -m research_engine ingest \
  /Users/iantodd/Projects/highdimensional/literature/coherence_time_corpus

python3 /Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/analysis/citation_corpus_audit.py \
  /Users/iantodd/Projects/highdimensional/literature/coherence_time_corpus/text
```

## How to use the report

Treat the report as triage, not proof.

- `Review Candidates` are the most interesting outputs.
- `Coverage Gaps` mean the cited paper is not in the shared text corpus yet.
- `Orphan References` often indicate broad conceptual citations or books, not necessarily mistakes.

Manual reading still matters for the highest-risk claims:

- mixing / recurrence claims
- cerebellum and motor timing claims
- psychedelic time-dilation claims
- ephaptic / field-substrate claims
