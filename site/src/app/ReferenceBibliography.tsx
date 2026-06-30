import { citationLabel, doiHref, fullReference, referenceSlug, type Reference } from "@/lib/audit";

export default function ReferenceBibliography({ references }: { references: Reference[] }) {
  const sorted = [...references].sort((a, b) => citationLabel(a).localeCompare(citationLabel(b)));

  return (
    <section className="reference-bibliography-section" aria-labelledby="references-heading">
      <h2 id="references-heading">References</h2>
      <ol className="reference-bibliography">
        {sorted.map((reference) => {
          const href = doiHref(reference.doi);
          const slug = referenceSlug(reference.cite_key);
          return (
            <li key={reference.cite_key} id={`ref-${slug}`}>
              <p>{fullReference(reference)}</p>
              <div className="reference-bibliography-links">
                <a href={`/references/${slug}`}>Native page</a>
                {href && (
                  <a href={href} target="_blank" rel="noreferrer">
                    DOI
                  </a>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
