import { notFound } from "next/navigation";
import { BookOpen, Database, ExternalLink } from "lucide-react";
import { citationLabel, doiHref, fullReference, referenceSlug, references, type Reference } from "@/lib/audit";
import { paperLibraryStatusText, referenceUseNote, uniqueReferenceContexts } from "@/lib/referenceUse";

type PageProps = {
  params: Promise<{ slug: string }>;
};

function referenceFromSlug(slug: string): Reference | null {
  return references.find((reference) => referenceSlug(reference.cite_key) === slug) ?? null;
}

export function generateStaticParams() {
  return references.map((reference) => ({ slug: referenceSlug(reference.cite_key) }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const reference = referenceFromSlug(slug);
  return {
    title: reference ? `${citationLabel(reference)} - cited reference` : "Cited reference",
  };
}

export default async function ReferenceDetailPage({ params }: PageProps) {
  const { slug } = await params;
  const reference = referenceFromSlug(slug);
  if (!reference) notFound();

  const href = doiHref(reference.doi);
  const note = referenceUseNote(reference);
  const contexts = uniqueReferenceContexts(reference, "plain");

  return (
    <main className="md-page reference-detail-page">
      <section className="reference-detail-hero">
        <div>
          <p className="md-eyebrow">{reference.cite_key}</p>
          <h1 className="md-h1">{reference.title}</h1>
          <p className="paper-subtitle">{citationLabel(reference)}</p>
          <p className="md-lead">{fullReference(reference)}</p>
          <div className="action-row">
            <a href="/" className="md-btn md-btn-filled">
              <BookOpen size={17} aria-hidden />
              Manuscript
            </a>
            {href && (
              <a href={href} className="md-btn md-btn-tonal" target="_blank" rel="noreferrer">
                DOI <ExternalLink size={15} aria-hidden />
              </a>
            )}
          </div>
        </div>
        <aside className="reference-status-panel">
          <p className="md-eyebrow">PaperLibrary</p>
          <strong>{reference.paperlibrary_status === "have_text" ? "Text harvested" : reference.paperlibrary_status ?? "Unknown"}</strong>
          <span>{reference.paperlibrary?.source ?? "site inventory"}</span>
          {reference.paperlibrary?.is_override && <span>Audit override</span>}
        </aside>
      </section>

      <section className="two-column reference-detail-columns">
        <article className="section-block">
          <p className="md-eyebrow">How the paper uses it</p>
          <h2 className="md-h2">{note.role}</h2>
          <div className="reference-explainer reference-explainer-standalone">
            <p>{note.what}</p>
          </div>
          <div className="reference-use-grid">
            <section>
              <p className="md-eyebrow">Why it matters</p>
              <p>{note.why}</p>
            </section>
            <section>
              <p className="md-eyebrow">Read with care</p>
              <p>{note.caution}</p>
            </section>
          </div>
          <h3 className="md-h3">{reference.citation_count} manuscript use{reference.citation_count === 1 ? "" : "s"}</h3>
          <ol className="reference-contexts">
            {contexts.map((context, index) => (
              <li key={`${reference.cite_key}-${index}`}>
                <span className="md-mono">context {index + 1}</span>
                <p>{context}</p>
              </li>
            ))}
          </ol>
        </article>

        <aside className="section-block">
          <p className="md-eyebrow">Local record</p>
          <h2 className="md-h2">Held source metadata</h2>
          <dl className="reference-record">
            <div>
              <dt>Authors</dt>
              <dd>{reference.authors || "Not recorded"}</dd>
            </div>
            <div>
              <dt>Journal/source</dt>
              <dd>{[reference.journal, reference.year].filter(Boolean).join(", ") || "Not recorded"}</dd>
            </div>
            <div>
              <dt>DOI</dt>
              <dd>{reference.doi || "Local or imported text"}</dd>
            </div>
            <div>
              <dt>PaperLibrary slug</dt>
              <dd>{reference.paperlibrary?.slug ?? "Not catalogued"}</dd>
            </div>
          </dl>
          <div className="reference-note">
            <Database size={18} aria-hidden />
            <p>{paperLibraryStatusText(reference)}</p>
          </div>
          {reference.paperlibrary?.note && (
            <div className="reference-note">
              <Database size={18} aria-hidden />
              <p>{reference.paperlibrary.note}</p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
