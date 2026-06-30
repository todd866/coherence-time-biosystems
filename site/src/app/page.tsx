
import { referencesByKey, type Reference } from "@/lib/audit";
import { loadNativeManuscript } from "@/lib/manuscript";
import NativePaperClient from "./NativePaperClient";

export default function ManuscriptHomePage() {
  const manuscript = loadNativeManuscript();
  const referenced = manuscript.citationKeys
    .map((key) => referencesByKey.get(key))
    .filter((reference): reference is Reference => Boolean(reference));

  return <NativePaperClient manuscript={manuscript} references={referenced} />;
}
