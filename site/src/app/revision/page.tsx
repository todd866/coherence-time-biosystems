
import { ArrowRight, Download, MessageSquareText } from "lucide-react";
import { paper, reframes, revisionPriorities, reviewerPosture } from "@/data/site";

export default function RevisionPage() {
  return (
    <main className="md-page">
      <section className="section-block">
        <p className="md-eyebrow">Paper map</p>
        <h1 className="md-h1">Argument and source map</h1>
        <p className="md-lead">{paper.shortThesis}</p>
        <div className="action-row"><a href="/paper.pdf" className="md-btn md-btn-filled"><Download size={17} aria-hidden />PDF</a><a href="/citation-use-audit.md" className="md-btn md-btn-tonal"><Download size={17} aria-hidden />Citation audit</a></div>
      </section>
      <section className="two-column posture-grid">{reviewerPosture.map((reviewer) => <article key={reviewer.reviewer} className="md-card posture-card"><div className="posture-head"><MessageSquareText size={21} aria-hidden /><div><p className="md-eyebrow">{reviewer.stance}</p><h2>{reviewer.reviewer}</h2></div></div><p>{reviewer.summary}</p></article>)}</section>
      <section className="section-block"><div className="section-head"><div><p className="md-eyebrow">Core reframing</p><h2 className="md-h2">What the paper changes</h2></div><span className="date-pill">{paper.manuscriptId}</span></div><div className="reframe-list">{reframes.map((item) => <article key={item.before} className="reframe-row"><p>{item.before}</p><ArrowRight size={18} aria-hidden /><p>{item.after}</p></article>)}</div></section>
      <section className="section-block"><p className="md-eyebrow">Action matrix</p><h2 className="md-h2">Highest-signal paper moves</h2><div className="dashboard-table-wrap"><table className="dashboard-table"><thead><tr><th>Move</th><th>Source</th><th>Priority</th><th>Purpose</th></tr></thead><tbody>{revisionPriorities.map((item) => <tr key={item.title}><td>{item.title}</td><td>{item.reviewer}</td><td><span className="badge badge-priority">{item.priority}</span></td><td>{item.body}</td></tr>)}</tbody></table></div></section>
    </main>
  );
}
