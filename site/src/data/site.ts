import {
          Activity, BookOpen, BrainCircuit, ClipboardList, Database, FileDown, Gauge, GitCompareArrows, Library, Milestone, Network, Radar, Route, Sparkles, Waves
        } from "lucide-react";

        const icons = {
          Activity, BookOpen, BrainCircuit, ClipboardList, Database, FileDown, Gauge, GitCompareArrows, Library, Milestone, Network, Radar, Route, Sparkles, Waves
        };

        export const paper = {
          title: "The Coherence Time of Intelligence",
          subtitle: "A phase-delta framework for temporal binding in high-dimensional biological computation",
          journal: "BioSystems",
          manuscriptId: "doi:10.1016/j.biosystems.2026.105755",
          decision: "Published",
          dueDate: "2026",
          doi: "10.1016/j.biosystems.2026.105755",
          abstract: "Commit times are bounded by how long semi-independent oscillatory modules must wait for alignment inside a tolerable phase window.",
          shortThesis: "Commit times are bounded by how long semi-independent oscillatory modules must wait for alignment inside a tolerable phase window.",
          mark: "C",
          pdfLabel: "PDF",
        };

        export const navLinks = [
          { href: "/", label: "Manuscript", icon: BookOpen },
          { href: "/dashboard", label: "Overview", icon: Radar },
          { href: "/revision", label: "Map", icon: ClipboardList },
          { href: "/audit", label: "Audit", icon: Gauge },
          { href: "/references", label: "References", icon: Library },
        ];

        export const downloads = [
  {
    "href": "/paper.pdf",
    "label": "PDF",
    "icon": "FileDown"
  },
  {
    "href": "/paper-source.tex",
    "label": "Source TeX",
    "icon": "FileDown"
  },
  {
    "href": "/citation-use-audit.md",
    "label": "Citation audit",
    "icon": "Database"
  },
  {
    "href": "/references.bib",
    "label": "Bibliography",
    "icon": "Library"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const headlineMetrics = [
  {
    "label": "Publication",
    "value": "BioSystems 262",
    "detail": "Article 105755 (2026)",
    "tone": "primary"
  },
  {
    "label": "Paper DOI",
    "value": "105755",
    "detail": "10.1016/j.biosystems.2026.105755",
    "tone": "tertiary"
  },
  {
    "label": "Cited sources",
    "value": "57",
    "detail": "57 bibliography entries parsed from TeX",
    "tone": "secondary"
  },
  {
    "label": "PDF/text coverage",
    "value": "52/57",
    "detail": "1 DOI-backed entries still need PaperLibrary harvest",
    "tone": "tertiary"
  }
];
        export const stanceCards = [
  {
    "title": "Phase-delta regime",
    "body": "Computation occurs in the approach to synchrony, before phase relationships collapse into a committed output.",
    "icon": "Waves"
  },
  {
    "title": "Coordination depth",
    "body": "Increasing the number of modules expands possible outcomes but makes alignment exponentially rarer.",
    "icon": "Network"
  },
  {
    "title": "Coherence-time bound",
    "body": "Commit timing follows a waiting-time law over the product manifold of inter-module phases.",
    "icon": "Route"
  },
  {
    "title": "Attention as speed-control",
    "body": "Attention can accelerate commits by increasing coherence and narrowing the coordination problem.",
    "icon": "Activity"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const revisionPriorities = [
  {
    "title": "Move from synchrony to approach",
    "reviewer": "Paper architecture",
    "priority": "Critical",
    "body": "Synchrony is treated as a commit event, not the whole computational state."
  },
  {
    "title": "Separate power from time",
    "reviewer": "Paper architecture",
    "priority": "High",
    "body": "Power bounds are non-binding in neural regimes; coordination time is the relevant constraint."
  },
  {
    "title": "Tie model to oscillation evidence",
    "reviewer": "Paper architecture",
    "priority": "High",
    "body": "The framework is grounded in working memory, attention, binding, and communication-through-coherence literature."
  },
  {
    "title": "Generate empirical predictions",
    "reviewer": "Paper architecture",
    "priority": "Medium",
    "body": "Alpha entrainment, task complexity, and attention should shift binding windows in quantifiable ways."
  }
];
        export const stressTests = [
  {
    "title": "Phase-delta regime",
    "body": "Computation occurs in the approach to synchrony, before phase relationships collapse into a committed output.",
    "icon": "Waves"
  },
  {
    "title": "Coordination depth",
    "body": "Increasing the number of modules expands possible outcomes but makes alignment exponentially rarer.",
    "icon": "Network"
  },
  {
    "title": "Coherence-time bound",
    "body": "Commit timing follows a waiting-time law over the product manifold of inter-module phases.",
    "icon": "Route"
  },
  {
    "title": "Attention as speed-control",
    "body": "Attention can accelerate commits by increasing coherence and narrowing the coordination problem.",
    "icon": "Activity"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const empiricalTests = [
  {
    "title": "Module count",
    "body": "Estimate how many semi-independent modules must align for a perceptual or cognitive commit."
  },
  {
    "title": "Baseline coherence",
    "body": "Measure the inter-module order parameter before and during task engagement."
  },
  {
    "title": "Commit timing",
    "body": "Compare predicted first-passage windows against behavioral and neural binding events."
  }
];
        export const evidenceClusters = [
  {
    "title": "Oscillatory cognition",
    "keys": [
      "miller2018",
      "lundqvist2016",
      "fries2015",
      "fries2005"
    ],
    "icon": "Waves"
  },
  {
    "title": "Timing and phase statistics",
    "keys": [
      "kuramoto1984",
      "mardia2000",
      "kac1947",
      "levinperes2017"
    ],
    "icon": "Route"
  },
  {
    "title": "Neural dimensionality",
    "keys": [
      "cunningham2014",
      "stringer2019",
      "amari2016"
    ],
    "icon": "BrainCircuit"
  },
  {
    "title": "Conscious timing tests",
    "keys": [
      "stetson2007",
      "samaha2015",
      "carhart2014",
      "casali2013"
    ],
    "icon": "Sparkles"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const reviewerPosture = [
  {
    "reviewer": "Published article",
    "stance": "BioSystems",
    "summary": "This native site renders the full published manuscript for Coherence Time as web text with live citations and local PDF exports."
  },
  {
    "reviewer": "Reference layer",
    "stance": "PaperLibrary-backed",
    "summary": "Each cited key receives a reference page with manuscript contexts, DOI links where available, and the current local PDF/text harvest state."
  }
];
        export const reframes = [
  {
    "before": "Synchrony is computation.",
    "after": "Synchrony terminates a computation by enabling commitment."
  },
  {
    "before": "Power sets the perceptual clock.",
    "after": "Coordination waiting time sets the relevant biological clock in cortical regimes."
  },
  {
    "before": "Fast cognition needs more energy.",
    "after": "Fast commits can also come from reduced module count or higher baseline coherence."
  }
];
        export const detailHighlights = [
  {
    "id": "main-thesis",
    "phrase": "Commit times are bounded by how long semi-independent oscillatory modules must wait for alignment inside a tolerable phase window.",
    "title": "Main thesis",
    "summary": "Commit times are bounded by how long semi-independent oscillatory modules must wait for alignment inside a tolerable phase window.",
    "bullets": [
      "Computation occurs in the approach to synchrony, before phase relationships collapse into a committed output.",
      "Increasing the number of modules expands possible outcomes but makes alignment exponentially rarer.",
      "Commit timing follows a waiting-time law over the product manifold of inter-module phases."
    ]
  }
];
