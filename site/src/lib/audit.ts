import inventory from "@/content/citation_inventory.json";

export type CitationContext = {
  line: number;
  context: string;
};

export type Reference = {
  cite_key: string;
  title: string;
  authors?: string;
  year?: string;
  journal?: string;
  doi?: string;
  citation_count: number;
  contexts: CitationContext[];
  paperlibrary_status?: string;
  paperlibrary?: {
    slug?: string;
    title?: string;
    year?: string;
    journal?: string;
    source?: string;
    pdf_path?: string;
    text_path?: string;
    note?: string;
    has_text?: boolean;
    is_override?: boolean;
  };
};

export type CitationInventory = {
  summary: {
    cited_keys: number;
    bib_entries: number;
    cited_dois: number;
    cited_without_doi: number;
    uncited_bib_entries: string[];
    missing_bib_entries: string[];
    paperlibrary_status_counts: Record<string, number>;
    text_overrides_used: string[];
  };
  references: Reference[];
};

const typedInventory = inventory as CitationInventory;

export const auditSummary = typedInventory.summary;

export const references = [...typedInventory.references].sort((a, b) => {
  const yearDelta = Number(b.year ?? 0) - Number(a.year ?? 0);
  if (yearDelta !== 0) return yearDelta;
  return a.cite_key.localeCompare(b.cite_key);
});

export const referencesByKey = new Map(references.map((reference) => [reference.cite_key, reference]));

export const topCitedReferences = [...references]
  .sort((a, b) => b.citation_count - a.citation_count || a.cite_key.localeCompare(b.cite_key))
  .slice(0, 12);

export const doiLessReferences = references.filter((reference) => !reference.doi);

export function doiHref(doi: string | undefined): string | null {
  if (!doi) return null;
  return `https://doi.org/${doi}`;
}

export function referenceSlug(key: string): string {
  return key
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function firstAuthor(reference: Pick<Reference, "authors" | "cite_key">): string {
  const author = reference.authors?.split(/\s+and\s+/)[0]?.trim();
  if (!author) return reference.cite_key;
  if (author.includes(",")) return author.split(",")[0].trim();
  return author.split(/\s+/).slice(-1)[0] ?? reference.cite_key;
}

export function citationLabel(reference: Reference): string {
  const authorCount = reference.authors?.split(/\s+and\s+/).filter(Boolean).length ?? 0;
  const suffix = authorCount > 1 ? " et al." : "";
  return [firstAuthor(reference) + suffix, reference.year].filter(Boolean).join(" ");
}

export function fullReference(reference: Reference): string {
  const source = [reference.journal, reference.year].filter(Boolean).join(", ");
  const doi = reference.doi ? `doi:${reference.doi}` : null;
  return [reference.authors, reference.title, source, doi].filter(Boolean).join(". ");
}

export function getClusterReferences(keys: string[]): Reference[] {
  return keys
    .map((key) => referencesByKey.get(key))
    .filter((reference): reference is Reference => Boolean(reference));
}
