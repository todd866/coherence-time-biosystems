# Citation Corpus Audit

This is a corpus-backed semantic audit. It compares citation contexts in `coherence_time.tex` against downloaded paper text when available, and falls back to auxiliary abstract/description text or bibliography-entry text when a paper has not been fully ingested yet.

## Run Info

- Backend: `sentence-transformers`
- Model: `all-MiniLM-L6-v2`
- Citation contexts analyzed: `57`
- Bibliography entries analyzed: `61`
- References with DOI: `57/61`
- References with extracted full text: `54/61`
- References with abstract/description text: `7/61`
- Title-only fallback references: `0/61`
- Contexts with at least one full-text-backed cited reference: `54/57`
- Top-1 cited match rate: `59.65%`
- Top-3 cited match rate: `75.44%`
- Mean best-cited similarity: `0.492`
- Median best-cited similarity: `0.493`

## How To Read This

- Strong alignments are useful mainly as a sanity check that the citation graph is coherent.
- Review candidates matter more: these are contexts where a cited paper has full text in the corpus but still looks semantically weak relative to the sentence.
- Coverage gaps are not citation errors. They mean the cited paper does not yet have full-text support in the corpus, so the audit is still relying on auxiliary or metadata-level fallback.

## Strongest Full-Text Alignments

- `coherence_time.tex:496` `CTX35` best cited full-text match `samaha2015` at `0.791`
  Context: Alpha oscillations (8 12 Hz) in visual cortex correlate with temporal acuity: individuals with faster alpha perceive time faster .
  Best chunk: --- Page 1 --- Report The Speed of Alpha-Band Oscillations Predicts the Temporal Resolution of Visual Perception Graphical Abstract Highlights d Individuals with higher alpha frequencies have vision with ﬁner temporal resolution d Eyes-closed and prestimulus peak alpha frequency both show this relationship d Within an 
- `coherence_time.tex:567` `CTX42` best cited full-text match `khrennikov2025bridge` at `0.752`
  Context: Quantum-like cognition. Recent work has explored connections between oscillatory neural networks and quantum-like cognitive phenomena .
  Best chunk: to QL behaviors. Inspired by the computational power of neuronal oscillations and quantum-inspired computation (QIC), we propose a quantum-theoretical framework for coupling of cognition/decision making and neural oscillations - QL oscillatory cognition. This is a step, may be very small, towards clarification of the r
- `coherence_time.tex:847` `CTX57` best cited full-text match `strogatz2000` at `0.679`
  Context: Relation to synchronisation theory. The Kuramoto model and its extensions provide a mature theory of synchronisation onset and order-parameter dynamics.
  Best chunk: crossed — then a small cluster of oscillators suddenly freezes into synchrony. This cooperative phenomenon apparently made a deep impression on Kuramoto. As he wrote in a paper with his student Nishikawa ([8], p. 570): “. . . Prigogine’s concept of time order [29], which refers to the spontaneous emergence of rhythms i
- `coherence_time.tex:49` `CTX01` best cited full-text match `lundqvist2016` at `0.731`
  Context: Working memory maintenance involves theta-gamma phase-amplitude coupling organized in discrete oscillatory bursts .
  Best chunk: Page 2 --- Neuron Article Gamma and Beta Bursts Underlie Working Memory Mikael Lundqvist,1,5 Jonas Rose,1,2,5 Pawel Herman,3 Scott L. Brincat,1 Timothy J. Buschman,1,4 and Earl K. Miller1,* 1The Picower Institute for Learning & Memory and Department of Brain & Cognitive Sciences, Massachusetts Institute of Technology (
- `coherence_time.tex:573` `CTX45` best cited full-text match `fries2005` at `0.717`
  Context: Communication through coherence (CTC). Fries's influential framework proposes that effective communication between neural populations requires coherence at the appropriate frequency: sending and receiving populations must be phase-aligned for spike-based signals to arrive during windows of maximal excitability.
  Best chunk: signals is governed solely by the structure of the anatomical connections, that is, there is no further communication structure beyond the one imposed by anatomical connectedness. However, cognitive functions require ﬂexibility in the routing of signals through the brain. They require a ﬂexible effective communication 
- `coherence_time.tex:709` `CTX49` best cited full-text match `cunningham2014` at `0.693`
  Context: Neural dynamics have measurable effective dimensionality. This is standard neuroscience: participation ratios, neural manifold analyses, and related methods quantify MATH from population recordings .
  Best chunk: variables are invariant to the scale of each neuron's activity61. Following these considerations, the data can be preprocessed by taking binned spike counts, averaging across trials and/or kernel smoothing across time33. Estimating and interpreting dimensionality Many dimensionality reduction methods require a choice o
- `coherence_time.tex:714` `CTX51` best cited full-text match `carhart2014` at `0.690`
  Context: Psychedelics increase neural signal complexity and produce subjectively enriched experience.
  Best chunk: ability to selectively target processes that appear to be critical for the maintenance of normal waking consciousness. In addressing the action of psychedelic drugs on the brain, this article begins at the cellular level before progressing to the systems level. The intention is to offer a comprehen- sive account of how
- `coherence_time.tex:480` `CTX34` best cited full-text match `stetson2007` at `0.675`
  Context: Critically, Stetson et al. showed that objective temporal resolution (tested via a falling LED chronometer) did not improve during free-fall fear only retrospective duration estimates increased.
  Best chunk: the tower and fall backward for 31 m before landing safely in a net below. doi:10.1371/journal.pone.0001295.g001 Figure 2. No evidence for fear-induced increase in temporal resolution. (a) Participants’ estimates of the duration of the free-fall were expanded by 36%. The actual duration of the fall was 2.49 sec. (b) If
- `coherence_time.tex:833` `CTX56` best cited full-text match `wolpert1998` at `0.584`
  Context: Operating with low coordination depth ( MATH 5), tight coupling ( MATH ), and stereotyped primitives, the cerebellum achieves commit times of 10 20 ms fast enough to override slow cortical dynamics when rapid action is required.
  Best chunk: models are found within the cerebellar circuitry. Forward internal models predict the consequences of actions and can be used to overcome time delays associated with feedback control. Secondly, we review the evidence that the cerebellum generates predictions using such a forward model. Finally, we review a computationa

## Review Candidates

- `coherence_time.tex:220` `CTX17` cited `cunningham2014, stringer2019`
  Context: The participation ratio provides an operational estimate: d eff = ( tr \, C) 2 tr (C 2) = ( i i ) 2 i i 2 where MATH are eigenvalues of MATH .
  Best cited full-text doc: `cunningham2014` at `0.096`
  Best cited chunk: Representative uses in neuroscience (refs.) PCA Covariance No No 6,29,50 FA Covariance No Yes 10,43,100 LDS/GPFA Dynamics Yes Yes 12,71,72 NLDS Dynamics Yes Yes 77,78 LDA Classification No No 9,19,56 Demixed Regression No Yes/No 4,16 Isomap/LLE Manifold discovery No No 14,44,58 Nat Neurosci. Author manuscript; availabl
  Best overall corpus match: `kac1947` at `0.210`
- `coherence_time.tex:679` `CTX48` cited `ashby1956`
  Context: A related control-theoretic point: Ashby's law of requisite variety implies that a subnetwork constrained to a low-dimensional regime loses the degrees of freedom needed to regulate coupled regions.
  Best cited full-text doc: `ashby1956` at `0.112`
  Best cited chunk: tries to show something of the wealth of pos- sible applications. It leads into Shannon’s theory; so after reading this Part the reader will be able to proceed without difficulty to the study of Shannon’s own work. Part III deals with mechanism and information as they are used in biological systems for regulation and c
  Best overall corpus match: `todd2025intelligence` at `0.314`
- `coherence_time.tex:752` `CTX52` cited `carhart2019rebus`
  Context: These increase MATH by relaxing precision weighting of top-down priors , loosening top-down constraints on dynamics normally suppressed by hierarchical filtering.
  Best cited full-text doc: `carhart2019rebus` at `0.137`
  Best cited chunk: and the precision (felt confidence) of posterior beliefs. The basic idea—pursued in this article—is that psychedelics act preferentially via stimulating 5-HT2ARs on deep pyramidal cells within the visual cortex as well as at higher levels of the cortical hierarchy. Deep-layer pyramidal neurons are thought to encode pos
  Best overall corpus match: `khrennikov2025bridge` at `0.332`
- `coherence_time.tex:784` `CTX54` cited `casali2013`
  Context: These likely decrease MATH by driving the cortex into lower-complexity, less integrative states .
  Best cited full-text doc: `casali2013` at `0.164`
  Best cited chunk: (iv) normalizing algorithmic complexity by the source entropy of SS(x,t) (28). Thus, operationally, PCI is defined as the normalized Lempel-Ziv complexity of the spatiotemporal pattern of cortical activation trig- gered by a direct TMS perturbation (see the Supplementary Materials for details of these steps). In practi
  Best overall corpus match: `khrennikov2025bridge` at `0.443`
- `coherence_time.tex:636` `CTX47` cited `miller2018`
  Context: The standard frequency-hierarchy view and the present framework make overlapping predictions in many cases.
  Best cited full-text doc: `miller2018` at `0.201`
  Best cited chunk: attention: ev- idence for discrete computations in cognition. Front. Hum. Neurosci. 4, 194. Buschman, T.J., Siegel, M., Roy, J.E., and Miller, E.K. (2011). Neural sub- strates of cognitive capacity limitations. Proc. Natl. Acad. Sci. USA 108, 11252–11255. Buschman, T.J., Denovellis, E.L., Diogo, C., Bullock, D., and Mi
  Best overall corpus match: `cecere2015` at `0.382`
- `coherence_time.tex:92` `CTX08` cited `amari2016`
  Context: Different measurement schemes (participation ratios, manifold dimension, information-geometric metrics) estimate this property from data .
  Best cited full-text doc: `amari2016` at `0.239`
  Best cited chunk: omissions that may have been made. Printed on acid-free paper This Springer imprint is published by SpringerNature The registered company is Springer Japan KK --- Page 6 --- Preface Information geometry is a method of exploring the world of information by means of modern geometry. Theories of information have so far be
  Best overall corpus match: `cunningham2014` at `0.323`
- `coherence_time.tex:49` `CTX04` cited `varela2001`
  Context: Long-range communication occurs preferentially during coherent states .
  Best cited full-text doc: `varela2001` at `0.313`
  Best cited chunk: perception1,33,68,99,100. Bottom-up and top-down are heuristic terms for what is in reality a large-scale network that integrates both incoming and endogenous activity; it is precisely at this level where phase synchronization is crucial as a mechanism for large-scale integration. Figure 1 | Schematic representation of
  Best overall corpus match: `fries2005` at `0.488`

## Coverage Gaps

- `coherence_time.tex:147` `CTX13` cited `mardia2000` has no full-text-backed cited reference yet
  Context: Let MATH denote the inter-module coherence MATH where MATH is the aggregate phase of module MATH . (We reserve MATH for the all-oscillator order parameter, which serves as a diagnostic but does not appear in Eq. .) The von Mises concentration MATH satisfies MATH .
- `coherence_time.tex:151` `CTX14` cited `mardia2000` has no full-text-backed cited reference yet
  Context: Why von Mises? Given circular symmetry and a fixed mean resultant length MATH , the maximum-entropy distribution on the circle is von Mises .
- `coherence_time.tex:510` `CTX38` cited `raichle2001` has no full-text-backed cited reference yet
  Context: Mind-wandering and creative insight (high MATH , low MATH ). Default-mode network activity is a canonical example of internally directed baseline brain activity .

## Orphan References

- `dyson2004` `description_text` max similarity `0.295`
  Title: A meeting with Enrico Fermi
- `gabriel1996` `full_text` max similarity `0.300`
  Title: The dielectric properties of biological tissues: III
- `ashby1956` `full_text` max similarity `0.337`
  Title: An Introduction to Cybernetics
- `pavey2024` `abstract_text` max similarity `0.415`
  Title: Distinct neuronal circuits mediate cortical hyperexcitability in amyotrophic lateral sclerosis
- `braitenberg1958` `full_text` max similarity `0.422`
  Title: Morphological observations on the cerebellar cortex
- `igamberdiev2025constraints` `full_text` max similarity `0.447`
  Title: Physical limits of natural computation as the biological constraints of morphogenesis, evolution, and consciousness: On the 100th anniversary of Efim Liberman (1925--2011)
- `llinas1982` `description_text` max similarity `0.469`
  Title: General discussion: Radial connectivity in the cerebellar cortex
- `womelsdorf2007` `full_text` max similarity `0.502`
  Title: Modulation of neuronal interactions through neuronal synchronization
- `ott2008` `full_text` max similarity `0.604`
  Title: Low dimensional behavior of large systems of globally coupled oscillators

## Probe Queries

- **Rare-event mixing support**
  Probe: rare event first passage times in rapidly mixing stochastic processes with recurrence and approximately exponential hitting times
  - `kac1947` `full_text` at `0.489`
  - `levinperes2017` `full_text` at `0.368`
  - `strogatz2000` `full_text` at `0.364`
- **Ephaptic timing support**
  Probe: extracellular electric fields modulate membrane potential and spike timing through ephaptic coupling
  - `subramanian2022` `full_text` at `0.619`
  - `han2018` `full_text` at `0.612`
  - `pinotsis2023` `full_text` at `0.578`
- **Psychedelic complexity support**
  Probe: psychedelics increase neural signal complexity and alter conscious experience
  - `carhart2014` `full_text` at `0.750`
  - `carhart2019rebus` `full_text` at `0.712`
  - `schartner2017` `full_text` at `0.679`
- **Consciousness complexity index**
  Probe: perturbational complexity index distinguishes conscious from unconscious states
  - `casali2013` `full_text` at `0.594`
  - `tononi2016` `full_text` at `0.534`
  - `khrennikov2021` `full_text` at `0.508`
- **Cerebellar fast motor control**
  Probe: cerebellum supports rapid motor primitives with low coordination depth and fast timing
  - `wolpert1998` `full_text` at `0.647`
  - `ito2008` `abstract_text` at `0.633`
  - `llinas1982` `description_text` at `0.508`
- **Entrainment timing prediction**
  Probe: transcranial alternating current stimulation entrains brain oscillations and shifts temporal processing
  - `helfrich2014` `full_text` at `0.770`
  - `subramanian2022` `full_text` at `0.470`
  - `engel2001` `full_text` at `0.460`

## Suggested Workflow

1. Ingest more cited papers into the shared literature store, or add auxiliary abstract/description text when full PDFs are unavailable.
2. Re-run this audit and focus on the review-candidate section, not the raw hit rate.
3. For any persistent mismatch, manually read the cited paper and either tighten the sentence or add a better source.

