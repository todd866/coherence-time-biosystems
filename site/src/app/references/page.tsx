import { auditSummary, references } from "@/lib/audit";
import ReferencesClient from "./ReferencesClient";

export default function ReferencesPage() {
  return (
    <main className="md-page">
      <section className="section-block">
        <p className="md-eyebrow">Reference layer</p>
        <h1 className="md-h1">Cited references</h1>
        <p className="md-lead">
          The R1 bibliography contains {auditSummary.cited_keys} cited keys, {auditSummary.cited_dois} DOI-backed
          sources, and {auditSummary.cited_without_doi} locally covered book or imported-text sources. Each citation now
          has a native web page for manuscript-use context and PaperLibrary status.
        </p>
        <div className="action-row">
          <a href="/references.bib" className="md-btn md-btn-filled">
            Download BibTeX
          </a>
          <a href="/citation-use-audit.md" className="md-btn md-btn-tonal">
            Audit notes
          </a>
        </div>
      </section>
      <ReferencesClient references={references} />
    </main>
  );
}
