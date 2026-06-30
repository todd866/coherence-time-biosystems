
import { ArrowRight, CalendarDays, CheckCircle2, FileText } from "lucide-react";
import { downloads, empiricalTests, headlineMetrics, paper, revisionPriorities, stanceCards, stressTests } from "@/data/site";
import { auditSummary } from "@/lib/audit";

function MetricCard({ metric }: { metric: (typeof headlineMetrics)[number] }) {
  return <article className={`metric-card metric-${metric.tone}`}><span>{metric.label}</span><strong>{metric.value}</strong><p>{metric.detail}</p></article>;
}

export default function OverviewPage() {
  return (
    <main className="md-page">
      <section className="overview-grid">
        <div className="overview-copy">
          <p className="md-eyebrow">Published paper surface</p>
          <h1 className="md-h1">{paper.title}</h1>
          <p className="paper-subtitle">{paper.subtitle}</p>
          <p className="md-lead">{paper.shortThesis}</p>
          <div className="action-row">
            <a href="/manuscript" className="md-btn md-btn-filled"><FileText size={17} aria-hidden />Manuscript</a>
            <a href="/audit" className="md-btn md-btn-tonal"><CheckCircle2 size={17} aria-hidden />Citation audit</a>
          </div>
        </div>
        <aside className="md-card native-paper-card" aria-label="Native paper surface">
          <p className="md-eyebrow">Native reading surface</p>
          <h2>Paper pages, not embedded PDFs</h2>
          <p>The manuscript is rendered as a web article with live citation panels and native reference pages. The PDF remains available as the formal published artifact.</p>
          <div className="native-preview-actions"><a href="/" className="md-btn md-btn-filled"><FileText size={16} aria-hidden />Read manuscript</a><a href="/references" className="md-btn md-btn-tonal">Reference pages</a></div>
        </aside>
      </section>
      <section className="metric-grid" aria-label="Paper status">{headlineMetrics.map((metric) => <MetricCard key={metric.label} metric={metric} />)}</section>
      <section className="section-block">
        <div className="section-head"><div><p className="md-eyebrow">Argument map</p><h2 className="md-h2">What the paper claims</h2></div><a href="/revision" className="section-link">Paper map <ArrowRight size={16} aria-hidden /></a></div>
        <div className="card-grid four">{stanceCards.map((card) => { const Icon = card.icon; return <article key={card.title} className="md-card stance-card"><Icon size={21} aria-hidden /><h3>{card.title}</h3><p>{card.body}</p></article>; })}</div>
      </section>
      <section className="two-column">
        <div className="section-block"><p className="md-eyebrow">Paper focus</p><h2 className="md-h2">Core moves</h2><div className="list-stack">{revisionPriorities.slice(0, 4).map((item) => <article key={item.title} className="list-item"><div><h3>{item.title}</h3><p>{item.body}</p></div><span className="badge badge-priority">{item.priority}</span></article>)}</div></div>
        <div className="section-block"><p className="md-eyebrow">Audit state</p><h2 className="md-h2">Evidence trail</h2><div className="audit-mini"><div><strong>{auditSummary.cited_keys}</strong><span>cited keys</span></div><div><strong>{auditSummary.bib_entries}</strong><span>bib entries</span></div><div><strong>{auditSummary.paperlibrary_status_counts.have_text ?? 0}</strong><span>texts harvested</span></div></div><p className="md-muted">The reference layer is generated from the manuscript bibliography and refreshed against the local PaperLibrary catalog.</p><a href="/audit" className="md-btn md-btn-outline">Open audit</a></div>
      </section>
      <section className="section-block"><div className="section-head"><div><p className="md-eyebrow">Stress tests</p><h2 className="md-h2">Where the argument has teeth</h2></div></div><div className="card-grid four">{stressTests.map((item) => { const Icon = item.icon; return <article key={item.title} className="md-card stance-card"><Icon size={21} aria-hidden /><h3>{item.title}</h3><p>{item.body}</p></article>; })}</div></section>
      <section className="section-block"><div className="section-head"><div><p className="md-eyebrow">Checks</p><h2 className="md-h2">Operational tests in the paper</h2></div><span className="date-pill"><CalendarDays size={15} aria-hidden />{paper.decision}</span></div><div className="test-list">{empiricalTests.map((test, index) => <article key={test.title} className="test-row"><span className="test-index">{index + 1}</span><div><h3>{test.title}</h3><p>{test.body}</p></div></article>)}</div></section>
      <section className="download-band" aria-label="Downloads">{downloads.map((download) => { const Icon = download.icon; return <a key={download.href} href={download.href} className="md-btn md-btn-tonal"><Icon size={16} aria-hidden />{download.label}</a>; })}</section>
    </main>
  );
}
