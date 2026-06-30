
"use client";

import dynamic from "next/dynamic";
import type { NativeManuscript } from "@/lib/manuscript";
import type { Reference } from "@/lib/audit";

type NativePaperClientProps = {
  manuscript: NativeManuscript;
  references: Reference[];
};

const NativePaperNoSSR = dynamic<NativePaperClientProps>(() => import("./NativePaper"), {
  ssr: false,
  loading: () => (
    <main className="paper-split">
      <div className="paper-main">
        <div className="paper-main-inner">
          <p className="md-eyebrow">Native manuscript</p>
          <h1 className="paper-title">Loading manuscript</h1>
        </div>
      </div>
    </main>
  ),
});

export default function NativePaperClient(props: NativePaperClientProps) {
  return <NativePaperNoSSR {...props} />;
}
