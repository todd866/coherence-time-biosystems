
import { citationLabel, type Reference } from "@/lib/audit";

export type ReferenceUseNote = {
  role: string;
  what: string;
  why: string;
  caution: string;
};

type CitationContextMode = "interactive" | "plain";

function citationText(keys: string, mode: CitationContextMode, parenthetical: boolean): string {
  const citeKeys = keys.split(",").map((key) => key.trim()).filter(Boolean);
  if (mode === "plain") return parenthetical ? `(${citeKeys.join(", ")})` : citeKeys.join(", ");
  return `[${citeKeys.map((key) => `@${key}`).join("; ")}]`;
}

export function cleanCitationContext(context: string, mode: CitationContextMode = "interactive"): string {
  return context
    .replace(/\\citet(?:\[[^\]]*\]){0,2}\{([^}]+)\}/g, (_, keys: string) => citationText(keys, mode, false))
    .replace(/\\citep(?:\[[^\]]*\]){0,2}\{([^}]+)\}/g, (_, keys: string) => citationText(keys, mode, true))
    .replace(/\\cite(?:\[[^\]]*\]){0,2}\{([^}]+)\}/g, (_, keys: string) => citationText(keys, mode, true))
    .replace(/\\(?:emph|textit|textbf)\{([^}]+)\}/g, "$1")
    .replace(/``|''/g, '"')
    .replace(/~/g, " ")
    .replace(/\\&/g, "&");
}

export function uniqueReferenceContexts(reference: Reference, mode: CitationContextMode = "interactive"): string[] {
  const seen = new Set<string>();
  const contexts: string[] = [];
  for (const context of reference.contexts) {
    const cleaned = cleanCitationContext(context.context, mode);
    const key = cleaned.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    contexts.push(cleaned);
  }
  return contexts;
}

export function referenceUseNote(reference: Reference): ReferenceUseNote {
  const title = `${reference.title} ${reference.journal ?? ""}`.toLowerCase();
  const firstContext = uniqueReferenceContexts(reference, "plain")[0] ?? "";
  if (/landauer|thermodynamic|entropy|demon|erasure|information/.test(title)) {
    return {
      role: "Thermodynamic constraint",
      what: `${citationLabel(reference)} supplies part of the physical-information background for the manuscript's measurement and registration claims.`,
      why: "The citation anchors the paper's move from abstract computation to energetically constrained biological records.",
      caution: "It should be read as a local constraint on the cited argument, not as a standalone proof of the paper's full framework.",
    };
  }
  if (/oscillat|synchron|coherence|phase|kuramoto|rhythm|neuron|cortex|brain|working memory/.test(title)) {
    return {
      role: "Neural dynamics evidence",
      what: `${citationLabel(reference)} helps establish that timing, phase, coherence, or population dynamics are active parts of the biological substrate.`,
      why: "The manuscript uses this literature to connect high-dimensional theory to measurable neural and biological dynamics.",
      caution: "The source supports the local substrate claim; the broader framework still depends on the surrounding derivation and comparison.",
    };
  }
  if (/dimension|manifold|geometry|bayesian|control|complexity|reachability|tractability/.test(title)) {
    return {
      role: "Dimensional or computational scaffold",
      what: `${citationLabel(reference)} supports the paper's treatment of dimensionality, tractability, control, or geometric structure.`,
      why: "The citation supplies precedent for treating dimensional structure as a causal and measurement-relevant quantity.",
      caution: "Dimensional estimates are method-dependent, so the manuscript uses the source as scaffolding for an argument rather than as a universal measurement.",
    };
  }
  if (/biosystems|biology|cell|protein|chemotaxis|bioelectric|conscious|free energy/.test(title)) {
    return {
      role: "Biological comparison point",
      what: `${citationLabel(reference)} places the manuscript's framework alongside a biological mechanism, case study, or adjacent theoretical account.`,
      why: "The reference helps keep the argument tied to biological systems rather than only to formal measurement theory.",
      caution: `The relevant claim is the way the source is used in context, especially in this passage: "${firstContext.slice(0, 220)}${firstContext.length > 220 ? "..." : ""}".`,
    };
  }
  return {
    role: "Supporting context",
    what: `${citationLabel(reference)} supplies background, precedent, or a comparison point for the local claim being made in the manuscript.`,
    why: "The native reference layer shows the exact manuscript contexts so the citation's argumentative role can be inspected directly.",
    caution: "Read the cited passage alongside the context list below; the page is meant to expose the local use rather than overstate the source.",
  };
}

export function paperLibraryStatusText(reference: Reference): string {
  if (reference.paperlibrary_status === "have_text") return "PaperLibrary has both a local PDF record and harvested text for this source.";
  if (reference.paperlibrary_status === "have_pdf") return "PaperLibrary has a local PDF record, but extracted text is not recorded for this source.";
  if (reference.paperlibrary_status === "needs_pdf") return "This DOI-backed source still needs to be pulled into PaperLibrary or reconciled with an existing local copy.";
  if (reference.paperlibrary_status === "no_doi") return "No DOI is recorded in the manuscript bibliography for this source; it needs title/book/manual handling rather than DOI harvest.";
  if (reference.paperlibrary_status) return `PaperLibrary status: ${reference.paperlibrary_status}.`;
  return "PaperLibrary status is not recorded for this source.";
}
