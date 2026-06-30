
import fs from "fs";
import path from "path";
import { paper } from "@/data/site";
import { firstAuthor, referencesByKey } from "@/lib/audit";
import figurePaths from "@/content/figure_paths.json";

export type NativeManuscript = {
  title: string;
  subtitle: string;
  body: string;
  citationKeys: string[];
  sectionCount: number;
};

const TEX_PATH = path.join(process.cwd(), "src", "content", "paper.tex");
const FIGURES = figurePaths as Record<string, string>;

function citationGroup(rawKeys: string): string {
  const keys = rawKeys.split(",").map((key) => key.trim()).filter(Boolean);
  return `[${keys.map((key) => `@${key}`).join("; ")}]`;
}

function textualCitation(rawKeys: string): string {
  const keys = rawKeys.split(",").map((key) => key.trim()).filter(Boolean);
  const first = referencesByKey.get(keys[0]);
  const lead = first ? firstAuthor(first) : keys[0];
  return `${lead} ${citationGroup(keys.join(","))}`;
}

function collectSectionLabels(tex: string): Map<string, number> {
  const labels = new Map<string, number>();
  const sectionPattern = /\\section\*?\{([^}]+)\}\s*\\label\{([^}]+)\}/g;
  let sectionNumber = 0;
  let match: RegExpExecArray | null;
  while ((match = sectionPattern.exec(tex)) !== null) {
    if (!match[0].startsWith("\\section*")) sectionNumber += 1;
    labels.set(match[2], sectionNumber);
  }
  return labels;
}

function stripLatex(value: string): string {
  let text = value.replace(/\n+/g, " ");
  text = text.replace(/\\href\{([^{}]+)\}\{([^{}]+)\}/g, "$2").replace(/\\url\{([^{}]+)\}/g, "$1");
  for (let guard = 0; guard < 10; guard += 1) {
    text = text.replace(/\\(?:emph|textbf|textit|texttt|mathrm|mathbf|mathcal)\{([^{}]+)\}/g, "$1");
  }
  return text.replace(/\\&/g, "&").replace(/\\%/g, "%").replace(/\\_/g, "_").replace(/[{}]/g, "").replace(/\s+/g, " ").trim();
}

function replaceSimpleCommands(markdown: string): string {
  let body = markdown;
  for (let guard = 0; guard < 14; guard += 1) {
    body = body
      .replace(/\\emph\{([^{}]+)\}/g, "*$1*")
      .replace(/\\textit\{([^{}]+)\}/g, "*$1*")
      .replace(/\\textbf\{([^{}]+)\}/g, "**$1**")
      .replace(/\\texttt\{([^{}]+)\}/g, "`$1`")
      .replace(/\\mathrm\{([^{}]+)\}/g, "$1")
      .replace(/\\mathbf\{([^{}]+)\}/g, "$1")
      .replace(/\\mathcal\{([^{}]+)\}/g, "$1");
  }
  return body;
}

function figureMarkdown(block: string): string {
  const include = block.match(/\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}/)?.[1];
  const captionRaw = block.match(/\\caption\{([\s\S]*?)\}\s*(?:\\label|\\end)/)?.[1] ?? "Figure";
  const caption = stripLatex(captionRaw);
  if (!include) return caption ? `\n\n**Figure.** ${caption}\n\n` : "\n\n";
  const src = FIGURES[include] ?? FIGURES[include.split("/").pop() ?? ""] ?? "";
  if (!src) return `\n\n**Figure.** ${caption}\n\n`;
  return `\n\n![${caption}](${src})\n\n*${caption}*\n\n`;
}

function tableMarkdown(block: string): string {
  const caption = stripLatex(block.match(/\\caption\{([\s\S]*?)\}/)?.[1] ?? "Table");
  const rows = block
    .replace(/\\caption\{[\s\S]*?\}/g, "")
    .replace(/\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}/g, "")
    .replace(/\\toprule|\\midrule|\\bottomrule|\\hline/g, "")
    .split(/\\\\/)
    .map((row) => stripLatex(row.replace(/&/g, " | ")))
    .filter((row) => row && !/^begin|^end/.test(row));
  return `\n\n**${caption}**\n\n\`\`\`text\n${rows.join("\n")}\n\`\`\`\n\n`;
}

function texToMarkdown(tex: string): string {
  const sectionLabels = collectSectionLabels(tex);
  let body = tex.split("\n").filter((line) => !line.trim().startsWith("%")).join("\n");
  body = body.replace(/^[\s\S]*?\\begin\{document\}/, "");
  body = body.replace(/\\begin\{thebibliography\}[\s\S]*?\\end\{thebibliography\}/g, "");
  body = body.replace(/\\end\{document\}[\s\S]*$/, "");
  body = body.replace(/\\maketitle/g, "");
  body = body.replace(/\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}/g, "");
  body = body.replace(/\\begin\{figure\}[\s\S]*?\\end\{figure\}/g, (block) => figureMarkdown(block));
  body = body.replace(/\\begin\{table\}[\s\S]*?\\end\{table\}/g, (block) => tableMarkdown(block));
  body = body
    .replace(/\\begin\{abstract\}/g, "## Abstract\n\n")
    .replace(/\\end\{abstract\}/g, "")
    .replace(/\\section\*\{([^}]+)\}/g, "## $1")
    .replace(/\\section\{([^}]+)\}/g, "## $1")
    .replace(/\\subsection\{([^}]+)\}/g, "### $1")
    .replace(/\\subsubsection\{([^}]+)\}/g, "#### $1")
    .replace(/\\paragraph\{([^}]+)\}/g, "#### $1")
    .replace(/\\label\{[^}]+\}/g, "")
    .replace(/Section~\\ref\{([^}]+)\}/g, (_, label: string) => {
      const section = sectionLabels.get(label);
      return section ? `Section ${section}` : "this section";
    })
    .replace(/\\ref\{([^}]+)\}/g, (_, label: string) => {
      const section = sectionLabels.get(label);
      return section ? String(section) : "this section";
    });
  body = body
    .replace(/\\citet(?:\[[^\]]*\]){0,2}\{([^}]+)\}/g, (_, keys: string) => textualCitation(keys))
    .replace(/\\citep(?:\[[^\]]*\]){0,2}\{([^}]+)\}/g, (_, keys: string) => citationGroup(keys))
    .replace(/\\cite(?:\[[^\]]*\]){0,2}\{([^}]+)\}/g, (_, keys: string) => citationGroup(keys));
  body = body
    .replace(/\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}/g, (_, eq: string) => `\n\n\`\`\`tex\n${eq.trim()}\n\`\`\`\n\n`)
    .replace(/\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}/g, (_, eq: string) => `\n\n\`\`\`tex\n${eq.trim()}\n\`\`\`\n\n`)
    .replace(/\\\[/g, "\n\n```tex\n")
    .replace(/\\\]/g, "\n```\n\n");
  body = replaceSimpleCommands(body);
  body = body
    .replace(/\\begin\{enumerate\}|\\end\{enumerate\}/g, "")
    .replace(/\\begin\{itemize\}|\\end\{itemize\}/g, "")
    .replace(/\\item\s+/g, "- ")
    .replace(/\$([^$\n]{1,160})\$/g, (_, expr: string) => `$${expr}$`)
    .replace(/``/g, '"')
    .replace(/''/g, '"')
    .replace(/\\noindent/g, "")
    .replace(/\\medskip|\\bigskip|\\smallskip/g, "")
    .replace(/\\&/g, "&")
    .replace(/\\%/g, "%")
    .replace(/\\_/g, "_")
    .replace(/~(?=[A-Za-z0-9\\])/g, " ")
    .replace(/\\begin\{[^}]+\}/g, "")
    .replace(/\\end\{[^}]+\}/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return body;
}

export function loadNativeManuscript(): NativeManuscript {
  const tex = fs.readFileSync(TEX_PATH, "utf8");
  const body = texToMarkdown(tex);
  const citationKeys = Array.from(new Set([...body.matchAll(/@([A-Za-z0-9_:-]+)/g)].map((match) => match[1]))).sort();
  const sectionCount = [...body.matchAll(/^##\s+/gm)].length;
  return { title: paper.title, subtitle: paper.subtitle, body, citationKeys, sectionCount };
}
