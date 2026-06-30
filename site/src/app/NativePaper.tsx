
"use client";

import { Fragment, Children, cloneElement, isValidElement, useEffect, useMemo, useRef, useState, type ReactElement, type ReactNode } from "react";
import { Database, FileDown, Library } from "lucide-react";
import { detailHighlights, paper } from "@/data/site";
import {
  auditSummary,
  citationLabel,
  doiHref,
  fullReference,
  referenceSlug,
  type Reference,
} from "@/lib/audit";
import type { NativeManuscript } from "@/lib/manuscript";
import { paperLibraryStatusText, referenceUseNote, uniqueReferenceContexts } from "@/lib/referenceUse";
import ReferenceBibliography from "./ReferenceBibliography";

type Active = { kind: "ref"; id: string } | { kind: "detail"; id: string } | null;
type DetailHighlight = { id: string; phrase: string; title: string; summary: string; bullets: string[] };
type InteractionContext = { refs: Map<string, Reference>; details: DetailHighlight[]; active: Active; onOpen: (active: Active) => void };

const CITE_RE = /\[@[A-Za-z0-9_:-]+(?:\s*;\s*@[A-Za-z0-9_:-]+)*\]/;

function citationKeys(group: string): string[] {
  return group.slice(1, -1).split(";").map((key) => key.trim().replace(/^@/, "")).filter(Boolean);
}

function withInteractions(children: ReactNode, ctx: InteractionContext, key = "k"): ReactNode {
  return Children.map(children, (child, index) => {
    if (typeof child === "string") return splitInteractiveText(child, ctx, `${key}-${index}`);
    if (isValidElement(child)) {
      const element = child as ReactElement<{ children?: ReactNode }>;
      if (element.props.children != null) return cloneElement(element, {}, withInteractions(element.props.children, ctx, `${key}-${index}`));
    }
    return child;
  });
}

function nextDetailMatch(text: string, details: DetailHighlight[]): { detail: DetailHighlight; index: number } | null {
  let best: { detail: DetailHighlight; index: number } | null = null;
  for (const detail of details) {
    const index = text.indexOf(detail.phrase);
    if (index === -1) continue;
    if (!best || index < best.index || (index === best.index && detail.phrase.length > best.detail.phrase.length)) best = { detail, index };
  }
  return best;
}

function splitInteractiveText(text: string, ctx: InteractionContext, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  let rest = text;
  let guard = 0;
  while (rest.length && guard < 800) {
    const citationMatch = rest.match(CITE_RE);
    const citationIndex = citationMatch?.index;
    const detailMatch = nextDetailMatch(rest, ctx.details);
    if ((citationIndex === undefined || citationIndex === -1) && !detailMatch) {
      out.push(rest);
      break;
    }
    const detailFirst = detailMatch && (citationIndex === undefined || detailMatch.index < citationIndex);
    if (detailFirst) {
      if (detailMatch.index > 0) out.push(rest.slice(0, detailMatch.index));
      const active = ctx.active?.kind === "detail" && ctx.active.id === detailMatch.detail.id;
      out.push(<button type="button" className={`detail-highlight${active ? " active" : ""}`} onClick={() => ctx.onOpen({ kind: "detail", id: detailMatch.detail.id })} key={`${key}-detail-${guard}`}>{detailMatch.detail.phrase}</button>);
      rest = rest.slice(detailMatch.index + detailMatch.detail.phrase.length);
      guard += 1;
      continue;
    }
    if (!citationMatch || citationIndex === undefined) {
      out.push(rest);
      break;
    }
    if (citationIndex > 0) out.push(rest.slice(0, citationIndex));
    out.push(<CitationGroup key={`${key}-cite-${guard}`} keys={citationKeys(citationMatch[0])} ctx={ctx} id={`${key}-cite-${guard}`} />);
    rest = rest.slice(citationIndex + citationMatch[0].length);
    guard += 1;
  }
  return out;
}

function CitationGroup({ keys, ctx, id }: { keys: string[]; ctx: InteractionContext; id: string }) {
  return <span className="cite" id={id}>{"("}{keys.map((refKey, index) => <Fragment key={`${id}-${refKey}-${index}`}><ReferenceInlineLink refKey={refKey} ctx={ctx} />{index < keys.length - 1 ? "; " : ""}</Fragment>)}{")"}</span>;
}

function ReferenceInlineLink({ refKey, ctx }: { refKey: string; ctx: InteractionContext }) {
  const reference = ctx.refs.get(refKey);
  const active = ctx.active?.kind === "ref" && ctx.active.id === refKey;
  if (!reference) return <span className="cite-link">{refKey}</span>;
  return <button type="button" className={`cite-link${active ? " active" : ""}`} onClick={() => ctx.onOpen({ kind: "ref", id: refKey })}>{citationLabel(reference)}</button>;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

type ReferenceMention = { refKey: string; index: number; end: number; length: number };

function referenceMentionSurfaces(refs: Map<string, Reference>): Array<{ refKey: string; surface: string }> {
  const surfaces: Array<{ refKey: string; surface: string }> = [];
  const seen = new Set<string>();
  for (const [refKey, reference] of refs) {
    for (const surface of [refKey, citationLabel(reference)]) {
      const normalized = surface.trim();
      const dedupeKey = normalized.toLowerCase();
      if (normalized.length < 4 || seen.has(dedupeKey)) continue;
      seen.add(dedupeKey);
      surfaces.push({ refKey, surface: normalized });
    }
  }
  return surfaces.sort((a, b) => b.surface.length - a.surface.length);
}

function nextReferenceMention(text: string, refs: Map<string, Reference>): ReferenceMention | null {
  let best: ReferenceMention | null = null;
  for (const candidate of referenceMentionSurfaces(refs)) {
    const pattern = new RegExp(`(^|[^A-Za-z0-9_:-])(${escapeRegExp(candidate.surface)})(?=$|[^A-Za-z0-9_:-])`, "i");
    const match = text.match(pattern);
    if (!match || match.index === undefined) continue;
    const prefixLength = match[1]?.length ?? 0;
    const matchedText = match[2];
    const index = match.index + prefixLength;
    const mention = { refKey: candidate.refKey, index, end: index + matchedText.length, length: matchedText.length };
    if (!best || mention.index < best.index || (mention.index === best.index && mention.length > best.length)) best = mention;
  }
  return best;
}

function splitSidebarText(text: string, ctx: InteractionContext, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  let rest = text;
  let guard = 0;
  while (rest.length && guard < 800) {
    const citationMatch = rest.match(CITE_RE);
    const citationIndex = citationMatch?.index;
    const refMention = nextReferenceMention(rest, ctx.refs);
    if ((citationIndex === undefined || citationIndex === -1) && !refMention) {
      out.push(rest);
      break;
    }
    const mentionFirst = refMention && (citationIndex === undefined || refMention.index < citationIndex);
    if (mentionFirst) {
      if (refMention.index > 0) out.push(rest.slice(0, refMention.index));
      out.push(<ReferenceInlineLink refKey={refMention.refKey} ctx={ctx} key={`${key}-ref-mention-${guard}`} />);
      rest = rest.slice(refMention.end);
      guard += 1;
      continue;
    }
    if (!citationMatch || citationIndex === undefined) {
      out.push(rest);
      break;
    }
    if (citationIndex > 0) out.push(rest.slice(0, citationIndex));
    out.push(<CitationGroup key={`${key}-cite-${guard}`} keys={citationKeys(citationMatch[0])} ctx={ctx} id={`${key}-cite-${guard}`} />);
    rest = rest.slice(citationIndex + citationMatch[0].length);
    guard += 1;
  }
  return out;
}

function linkSidebarReferences(children: ReactNode, ctx: InteractionContext, key = "sidebar"): ReactNode {
  return Children.map(children, (child, index) => {
    if (typeof child === "string") return splitSidebarText(child, ctx, `${key}-${index}`);
    if (isValidElement(child)) {
      const element = child as ReactElement<{ children?: ReactNode }>;
      if (element.props.children != null) return cloneElement(element, {}, linkSidebarReferences(element.props.children, ctx, `${key}-${index}`));
    }
    return child;
  });
}

function ReferenceUsePanel({ reference, ctx }: { reference: Reference; ctx: InteractionContext }) {
  const note = referenceUseNote(reference);
  const contexts = uniqueReferenceContexts(reference);
  return (
    <>
      <section className="reference-explainer">
        <p className="reference-role">{note.role}</p>
        <p>{linkSidebarReferences(note.what, ctx)}</p>
      </section>
      <DetailLayer label="Why it matters here" ctx={ctx}><p>{note.why}</p></DetailLayer>
      <DetailLayer label="How to read the citation" ctx={ctx}><p>{note.caution}</p></DetailLayer>
      {contexts.length > 0 && <DetailLayer label="Where it appears" ctx={ctx}><ol className="context-list compact">{contexts.slice(0, 3).map((context, index) => <li key={`${reference.cite_key}-context-${index}`}>{context}</li>)}</ol></DetailLayer>}
    </>
  );
}

function DetailLayer({ label, children, ctx }: { label: string; children: ReactNode; ctx?: InteractionContext }) {
  if (!children) return null;
  return <div className="paper-layer"><p className="md-eyebrow">{label}</p><div>{ctx ? linkSidebarReferences(children, ctx) : children}</div></div>;
}

function formatMathText(value: string): string {
  let out = value.trim();
  out = out
    .replace(/^\$|\$$/g, "")
    .replace(/\\(?:mathrm|mathbf|mathcal|operatorname)\{([^{}]+)\}/g, "$1")
    .replace(/\\mathbb\{R\}/g, "ℝ")
    .replace(/\\mathbb\{N\}/g, "ℕ")
    .replace(/\\mathbb\{Z\}/g, "ℤ")
    .replace(/\\mathbb\{C\}/g, "ℂ")
    .replace(/\\to|\\rightarrow/g, "→")
    .replace(/\\mapsto/g, "↦")
    .replace(/\\leftarrow/g, "←")
    .replace(/\\leftrightarrow/g, "↔")
    .replace(/\\leq?|\\le/g, "≤")
    .replace(/\\geq?|\\ge/g, "≥")
    .replace(/\\neq|\\ne/g, "≠")
    .replace(/\\approx/g, "≈")
    .replace(/\\sim/g, "∼")
    .replace(/\\propto/g, "∝")
    .replace(/\\infty/g, "∞")
    .replace(/\\in/g, "∈")
    .replace(/\\notin/g, "∉")
    .replace(/\\subseteq/g, "⊆")
    .replace(/\\subset/g, "⊂")
    .replace(/\\cup/g, "∪")
    .replace(/\\cap/g, "∩")
    .replace(/\\times/g, "×")
    .replace(/\\cdot/g, "·")
    .replace(/\\pm/g, "±")
    .replace(/\\ldots|\\dots/g, "…")
    .replace(/\\Delta/g, "Δ")
    .replace(/\\delta/g, "δ")
    .replace(/\\epsilon|\\varepsilon/g, "ε")
    .replace(/\\theta/g, "θ")
    .replace(/\\lambda/g, "λ")
    .replace(/\\mu/g, "μ")
    .replace(/\\sigma/g, "σ")
    .replace(/\\phi|\\varphi/g, "φ")
    .replace(/\\Phi/g, "Φ")
    .replace(/\\Omega/g, "Ω")
    .replace(/\\omega/g, "ω")
    .replace(/\\alpha/g, "α")
    .replace(/\\beta/g, "β")
    .replace(/\\gamma/g, "γ")
    .replace(/\\dim/g, "dim")
    .replace(/\\log/g, "log")
    .replace(/\\exp/g, "exp")
    .replace(/\\sqrt\{([^{}]+)\}/g, "√($1)")
    .replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1)/($2)")
    .replace(/\\\{/g, "{")
    .replace(/\\\}/g, "}")
    .replace(/\\,/g, " ")
    .replace(/\\;/g, " ")
    .replace(/\\:/g, " ")
    .replace(/\\!/g, "")
    .replace(/[{}]/g, "")
    .replace(/\s+/g, " ");
  return out.trim();
}

function inlineMarkdown(text: string, ctx: InteractionContext, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const tokenPattern = /(\$[^$\n]+\$|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > cursor) out.push(<Fragment key={`${key}-text-${index}`}>{withInteractions(text.slice(cursor, match.index), ctx, `${key}-text-${index}`)}</Fragment>);
    const token = match[0];
    const inner = token.replace(/^`|`$/g, "").replace(/^\*\*|\*\*$/g, "").replace(/^\*|\*$/g, "");
    if (token.startsWith("$")) out.push(<span className="math-inline" key={`${key}-math-${index}`}>{formatMathText(token)}</span>);
    else if (token.startsWith("`")) out.push(<code key={`${key}-code-${index}`}>{inner}</code>);
    else if (token.startsWith("**")) out.push(<strong key={`${key}-strong-${index}`}>{withInteractions(inner, ctx, `${key}-strong-${index}`)}</strong>);
    else out.push(<em key={`${key}-em-${index}`}>{withInteractions(inner, ctx, `${key}-em-${index}`)}</em>);
    cursor = match.index + token.length;
    index += 1;
  }
  if (cursor < text.length) out.push(<Fragment key={`${key}-text-tail`}>{withInteractions(text.slice(cursor), ctx, `${key}-text-tail`)}</Fragment>);
  return out;
}

function renderMarkdownBlock(block: string, index: number, ctx: InteractionContext): ReactNode {
  const text = block.trim();
  if (!text) return null;
  const heading = text.match(/^(#{2,4})\s+(.+)$/);
  if (heading) {
    const level = heading[1].length;
    const children = inlineMarkdown(heading[2], ctx, `h-${index}`);
    if (level === 2) return <h2 key={`h-${index}`}>{children}</h2>;
    if (level === 3) return <h3 key={`h-${index}`}>{children}</h3>;
    return <h4 key={`h-${index}`}>{children}</h4>;
  }
  if (text.startsWith("```")) {
    const lines = text.split("\n");
    const lang = lines[0].replace(/^```/, "").trim();
    const code = lines.slice(1, lines.at(-1)?.startsWith("```") ? -1 : undefined).join("\n");
    if (lang === "tex") return <div className="math-display" key={`math-${index}`}>{code.split(/\n+/).map((line, lineIndex) => <span key={`math-${index}-${lineIndex}`}>{formatMathText(line.replace(/&/g, "").replace(/\\\\$/g, ""))}</span>)}</div>;
    return <pre key={`code-${index}`}><code>{code}</code></pre>;
  }
  const image = text.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
  if (image) return <figure key={`fig-${index}`}><img src={image[2]} alt={image[1] || "Manuscript figure"} /></figure>;
  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  if (lines.length > 1 && lines.every((line) => line.startsWith("- "))) {
    return <ul key={`ul-${index}`}>{lines.map((line, lineIndex) => <li key={`li-${index}-${lineIndex}`}>{inlineMarkdown(line.slice(2), ctx, `li-${index}-${lineIndex}`)}</li>)}</ul>;
  }
  return <p key={`p-${index}`}>{inlineMarkdown(text.replace(/\s*\n\s*/g, " "), ctx, `p-${index}`)}</p>;
}

function renderManuscript(body: string, ctx: InteractionContext): ReactNode[] {
  return body.split(/\n{2,}/).map((block, index) => renderMarkdownBlock(block, index, ctx)).filter(Boolean);
}

export default function NativePaper({ manuscript, references }: { manuscript: NativeManuscript; references: Reference[] }) {
  const [active, setActive] = useState<Active>(null);
  const asideRef = useRef<HTMLElement>(null);
  const refs = useMemo(() => new Map(references.map((reference) => [reference.cite_key, reference])), [references]);
  const activeRef = active?.kind === "ref" ? refs.get(active.id) ?? null : null;
  const activeDetail = active?.kind === "detail" ? detailHighlights.find((detail) => detail.id === active.id) ?? null : null;
  useEffect(() => { if (asideRef.current) asideRef.current.scrollTop = 0; }, [active]);
  const interactionContext = { refs, details: detailHighlights, active, onOpen: setActive };

  return (
    <main className="paper-split">
      <div className="paper-main">
        <div className="paper-main-inner">
          <p className="md-eyebrow">Native manuscript</p>
          <h1 className="paper-title">{manuscript.title}</h1>
          <p className="paper-subtitle">{manuscript.subtitle}</p>
          <div className="paper-doc-actions">
            <a href="/paper.pdf" className="md-btn md-btn-filled"><FileDown size={16} aria-hidden />{paper.pdfLabel}</a>
            <a href="/paper-source.tex" className="md-btn md-btn-tonal"><FileDown size={16} aria-hidden />Source TeX</a>
            <a href="/references" className="md-btn md-btn-tonal"><Library size={16} aria-hidden />References</a>
          </div>
          <article className="paper">
            {renderManuscript(manuscript.body, interactionContext)}
            <ReferenceBibliography references={references} />
          </article>
        </div>
      </div>
      <aside ref={asideRef} className={`paper-aside${activeRef || activeDetail ? " open" : ""}`}>
        {activeRef ? (
          <>
            <div className="paper-panel-head"><p className="md-eyebrow">Reference</p><button type="button" className="md-btn md-btn-text" onClick={() => setActive(null)} aria-label="Close reference panel">x</button></div>
            <h2>{citationLabel(activeRef)}</h2>
            <p className="reference-full">{linkSidebarReferences(fullReference(activeRef), interactionContext)}</p>
            <div className="paper-panel-actions">
              <a href={`/references/${referenceSlug(activeRef.cite_key)}`} className="md-btn md-btn-filled" target="_blank" rel="noreferrer">Native page</a>
              {doiHref(activeRef.doi) && <a href={doiHref(activeRef.doi) ?? undefined} className="md-btn md-btn-tonal" target="_blank" rel="noreferrer">DOI</a>}
            </div>
            <ReferenceUsePanel reference={activeRef} ctx={interactionContext} />
            <DetailLayer label="PaperLibrary" ctx={interactionContext}><p>{paperLibraryStatusText(activeRef)}</p>{activeRef.paperlibrary?.note && <p>{activeRef.paperlibrary.note}</p>}</DetailLayer>
          </>
        ) : activeDetail ? (
          <>
            <div className="paper-panel-head"><p className="md-eyebrow">More detail</p><button type="button" className="md-btn md-btn-text" onClick={() => setActive(null)} aria-label="Close detail panel">x</button></div>
            <h2>{activeDetail.title}</h2>
            <p className="reference-full">{linkSidebarReferences(activeDetail.summary, interactionContext)}</p>
            <DetailLayer label="Why it matters" ctx={interactionContext}><ul className="detail-list">{activeDetail.bullets.map((bullet, index) => <li key={`${activeDetail.id}-bullet-${index}`}>{bullet}</li>)}</ul></DetailLayer>
          </>
        ) : (
          <div className="paper-panel-default">
            <p className="md-eyebrow">The working</p>
            <p>This is the native web manuscript. Every citation is live, and the PDF is a downloadable export rather than the reading surface.</p>
            <div className="paper-panel-stats">
              <div><strong>{auditSummary.cited_keys}</strong><span>cited keys</span></div>
              <div><strong>{auditSummary.paperlibrary_status_counts.have_text ?? 0}</strong><span>harvested texts</span></div>
              <div><strong>{manuscript.sectionCount}</strong><span>native sections</span></div>
            </div>
            <a href="/audit" className="md-btn md-btn-tonal"><Database size={16} aria-hidden />Audit layer</a>
          </div>
        )}
      </aside>
    </main>
  );
}
