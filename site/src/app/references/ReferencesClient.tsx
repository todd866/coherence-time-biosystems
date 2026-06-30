"use client";

import { BookOpen, ExternalLink, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { referenceSlug, type Reference } from "@/lib/audit";

type Filter = "all" | "doi" | "local" | "high-use";

function doiHref(doi: string | undefined): string | null {
  if (!doi) return null;
  return `https://doi.org/${doi}`;
}

export default function ReferencesClient({ references }: { references: Reference[] }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return references.filter((reference) => {
      if (filter === "doi" && !reference.doi) return false;
      if (filter === "local" && reference.doi) return false;
      if (filter === "high-use" && reference.citation_count < 3) return false;
      if (!q) return true;
      return [
        reference.cite_key,
        reference.title,
        reference.authors ?? "",
        reference.journal ?? "",
        reference.year ?? "",
        reference.doi ?? "",
      ]
        .join(" ")
        .toLowerCase()
        .includes(q);
    });
  }, [filter, query, references]);

  const filters: { value: Filter; label: string }[] = [
    { value: "all", label: "All" },
    { value: "doi", label: "DOI" },
    { value: "local", label: "Local text" },
    { value: "high-use", label: "High-use" },
  ];

  return (
    <>
      <section className="reference-controls">
        <label className="search-box">
          <Search size={17} aria-hidden />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search key, title, author, DOI" />
        </label>
        <div className="segmented-control" aria-label="Reference filter">
          {filters.map((item) => (
            <button key={item.value} type="button" className={filter === item.value ? "active" : ""} onClick={() => setFilter(item.value)}>
              {item.label}
            </button>
          ))}
        </div>
      </section>

      <p className="reference-count">
        Showing {filtered.length} of {references.length} cited references.
      </p>

      <section className="references-grid" aria-label="Cited references">
        {filtered.map((reference) => {
          const href = doiHref(reference.doi);
          return (
            <article key={reference.cite_key} className="md-card reference-card">
              <div className="reference-card-main">
                <p className="md-eyebrow">{reference.cite_key}</p>
                <h2>{reference.title}</h2>
                <p className="reference-card-meta">
                  {[reference.year, reference.journal].filter(Boolean).join(" - ")}
                </p>
                <p>{reference.authors}</p>
              </div>
              <div className="reference-card-actions">
                <span className="badge badge-variant">{reference.citation_count} uses</span>
                <a href={`/references/${referenceSlug(reference.cite_key)}`} className="md-btn md-btn-filled">
                  <BookOpen size={15} aria-hidden />
                  Native page
                </a>
                {href ? (
                  <a href={href} className="md-btn md-btn-tonal" target="_blank" rel="noreferrer">
                    DOI <ExternalLink size={15} aria-hidden />
                  </a>
                ) : (
                  <span className="badge badge-local">local text</span>
                )}
              </div>
            </article>
          );
        })}
      </section>
    </>
  );
}
