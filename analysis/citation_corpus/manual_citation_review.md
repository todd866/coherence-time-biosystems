# Manual Citation Review

This note records a human pass over the strongest embedding-space flags after
importing the available citation PDFs into the shared corpus.

The corpus-backed audit is useful as triage, not proof. A low semantic score can
mean either:

- a real support mismatch
- a broad conceptual citation where the sentence mostly contains the paper's own
  framing rather than the cited author's wording
- a formula-heavy line that embeddings handle badly

## Current coverage

- Full-text-backed references: `50/61`
- Citation contexts with at least one full-text-backed cited reference: `49/57`
- Remaining unsourced external refs: `8`

## Real Reviewer-Risk Items

### 1. `casali2013` on anaesthetics

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:784`
- Corpus finding: cited full-text similarity was essentially zero
- Judgment: real over-attribution risk

Reason:

`casali2013` supports reduced perturbational complexity / reduced capacity for
conscious integrated dynamics under anaesthesia. It does **not** directly support
the stronger phrasing that anaesthetics "fragment large-scale field coherence."

Action taken:

- Softened the sentence so `casali2013` now supports lower-complexity, less
  integrative states, with field-coherence fragmentation presented as a plausible
  accompanying mechanism rather than the cited paper's direct claim.

### 2. `miller2018` on the "frequency-hierarchy view"

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:636`
- Corpus finding: low cited similarity; best-matching chunk was about working
  memory rhythms, not anaesthesia
- Judgment: real attribution stretch in the old wording

Reason:

`miller2018` supports a rhythm-structured view of working memory and cortical
 computation. It does **not** specifically anchor the clause that "both agree
 that anaesthesia reduces dimensionality."

Action taken:

- Rephrased the sentence to avoid attributing the anaesthesia clause directly to
  `miller2018`.

## Low-Score Flags That Look Acceptable

### `carhart2019rebus`

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:751`
- Judgment: acceptable

Reason:

The full text explicitly discusses relaxation of the precision of high-level
priors and deep pyramidal-cell / hierarchical mechanisms. The low score is due
to conceptual wording, not lack of support.

### `tononi2016`

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:569`
- Judgment: acceptable after wording change

Reason:

The sentence is in a "relationship to existing frameworks" section. `tononi2016`
is there to identify IIT, while the thermodynamic/temporal constraint claim is
the present paper's contribution. The original wording risked sounding like
Tononi supplied those foundations directly, so the sentence was tightened.

### `ashby1956`

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:678`
- Judgment: acceptable as a conceptual citation

Reason:

This is a control-theoretic analogy built on requisite variety, not a claim that
Ashby discussed neural dimensionality in modern terms. Low lexical overlap is
expected.

### `varela2001`

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:49`
- Judgment: acceptable

Reason:

`varela2001` is clearly about phase synchronization and large-scale integration.
`fries2005` is a more direct communication-through-coherence citation, but the
Varela sentence is still defensible.

### `amari2016`

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:92`
- Judgment: acceptable but broad

Reason:

The citation is only for information-geometric metrics as one estimator family.
The sentence is broad and does not claim Amari provides all listed estimators.

### `cunningham2014,stringer2019`

- Manuscript location: `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/coherence_time.tex:220`
- Judgment: acceptable, likely an embedding false positive

Reason:

This is a formula-heavy line. The cited papers support dimensionality reduction
and neural effective dimensionality in practice, even if they do not present this
exact equation in the sentence's notation.

## Remaining Citation-Corpus Gaps

These still lack full-text support in the current corpus and remain worth
tracking:

- `mardia2000`
- `kuramoto1984`
- `raichle2001`
- `dyson2004`
- `ito2008`
- `shew2013`
- `llinas1982`
- `pavey2024`

The current sourcing checklist is:

- `/Users/iantodd/Projects/highdimensional/biosystems/4_coherence_time/analysis/citation_corpus/papers_to_source.md`
- `/Users/iantodd/Desktop/coherence_time_papers_to_source.html`

## Bottom Line

The embedding-space experiment became materially more useful once the PDFs were
imported. After human review, the main manuscript-level citation risks were

- over-attributing `casali2013`
- over-attributing `miller2018`

Those two points have now been tightened in the manuscript. The remaining
low-score flags are mostly broad conceptual citations or formula-heavy false
positives rather than obvious citation errors.
