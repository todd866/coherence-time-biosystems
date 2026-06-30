
import { CheckCircle2, Database, Library, SearchCheck } from "lucide-react";
import { evidenceClusters } from "@/data/site";
import { auditSummary, doiHref, doiLessReferences, getClusterReferences, references, topCitedReferences } from "@/lib/audit";

export default function AuditPage() {
  const textCoverage = auditSummary.paperlibrary_status_counts.have_text ?? 0;
  const needsPdf = auditSummary.paperlibrary_status_counts.needs_pdf ?? 0;
  const doiDenom = auditSummary.cited_dois + auditSummary.cited_without_doi;
  const doiCoveragePct = doiDenom ? Math.round((auditSummary.cited_dois / doiDenom) * 100) : 0;
  const paperLibraryPct = auditSummary.bib_entries ? Math.round((textCoverage / auditSummary.bib_entries) * 100) : 0;
  const doiBacked = references.filter((reference) => reference.doi).length;
  return (
    <main className="md-page">
      <section className="section-block"><p className="md-eyebrow">Citation audit</p><h1 className="md-h1">PaperLibrary and citation-use state</h1><p className="md-lead">The citation layer is generated from the paper's inline bibliography, linked back to manuscript contexts, and reconciled with the local PaperLibrary catalog.</p><div className="action-row"><a href="/citation-use-audit.md" className="md-btn md-btn-filled"><Database size={17} aria-hidden />Audit report</a><a href="/references" className="md-btn md-btn-tonal"><SearchCheck size={17} aria-hidden />Browse references</a></div></section>
      <section className="metric-grid audit-metrics" aria-label="Audit metrics">
        <article className="metric-card metric-success"><span>PaperLibrary text</span><strong>{textCoverage}/{auditSummary.bib_entries}</strong><p>References with local harvested text in the catalog.</p></article>
        <article className="metric-card metric-secondary"><span>DOI-backed entries</span><strong>{doiBacked}</strong><p>Entries with DOI metadata or DOI links in the bibliography.</p></article>
        <article className="metric-card metric-primary"><span>Cited keys</span><strong>{auditSummary.cited_keys}</strong><p>Manuscript citation keys with context panels.</p></article>
        <article className="metric-card metric-tertiary"><span>Needs harvest</span><strong>{needsPdf}</strong><p>DOI-backed entries not yet present as local PaperLibrary PDFs.</p></article>
      </section>
      <section className="section-block"><p className="md-eyebrow">Coverage bars</p><h2 className="md-h2">Where the audit stands</h2><div className="bar-list"><div className="bar-row"><div><strong>DOI-backed cited references</strong><span>{doiCoveragePct}% of cited references carry DOI metadata</span></div><div className="progress-bar" aria-label="DOI coverage"><span style={{ width: `${doiCoveragePct}%` }} /></div></div><div className="bar-row"><div><strong>PaperLibrary harvested text</strong><span>{paperLibraryPct}% of bibliography entries have extracted text</span></div><div className="progress-bar" aria-label="PaperLibrary coverage"><span style={{ width: `${paperLibraryPct}%` }} /></div></div><div className="bar-row"><div><strong>Reference pages</strong><span>Every parsed bibliography entry receives a native reference page</span></div><div className="progress-bar" aria-label="Reference pages"><span style={{ width: "100%" }} /></div></div></div></section>
      <section className="section-block"><div className="section-head"><div><p className="md-eyebrow">Evidence clusters</p><h2 className="md-h2">How the reference set supports the paper</h2></div></div><div className="card-grid two">{evidenceClusters.map((cluster) => { const Icon = cluster.icon; const clusterRefs = getClusterReferences(cluster.keys); return <article key={cluster.title} className="md-card cluster-card"><Icon size={21} aria-hidden /><h3>{cluster.title}</h3><ul>{clusterRefs.map((reference) => <li key={reference.cite_key}><span className="md-mono">{reference.cite_key}</span><span>{reference.title}</span></li>)}</ul></article>; })}</div></section>
      <section className="section-block"><p className="md-eyebrow">Most used citations</p><h2 className="md-h2">References carrying repeated argumentative load</h2><div className="dashboard-table-wrap"><table className="dashboard-table"><thead><tr><th>Key</th><th>Title</th><th>Year</th><th>Uses</th><th>DOI</th></tr></thead><tbody>{topCitedReferences.map((reference) => { const href = doiHref(reference.doi); return <tr key={reference.cite_key}><td className="md-mono">{reference.cite_key}</td><td>{reference.title}</td><td>{reference.year}</td><td>{reference.citation_count}</td><td>{href ? <a href={href}>DOI</a> : "local/manual"}</td></tr>; })}</tbody></table></div></section>
      <section className="section-block"><p className="md-eyebrow">Manual tail</p><h2 className="md-h2">Sources requiring non-DOI handling</h2><div className="caveat-grid"><article className="caveat-row"><Library size={20} aria-hidden /><p>{doiLessReferences.length} bibliography entries have no DOI in the manuscript bibliography: {doiLessReferences.map((reference) => reference.cite_key).join(", ") || "none"}.</p></article><article className="caveat-row"><CheckCircle2 size={20} aria-hidden /><p>DOI-backed entries can be bulk-pulled through PaperLibrary. Books, white papers, web pages, and DOI-less articles need manual import or title-based resolution.</p></article></div></section>
    </main>
  );
}
