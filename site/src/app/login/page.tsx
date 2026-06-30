
"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { LockKeyhole } from "lucide-react";
import { paper } from "@/data/site";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  async function handleSubmit(e: FormEvent) {
    e.preventDefault(); setError(""); setLoading(true);
    try {
      const res = await fetch("/api/auth", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password }) });
      const data = await res.json();
      if (data.ok) router.push("/"); else setError("That password did not match. Try again.");
    } catch { setError("Something went wrong. Please try again."); }
    finally { setLoading(false); }
  }
  return (
    <main className="login-shell"><div className="md-card login-card"><div className="login-mark" aria-hidden>{paper.mark}</div><h1>{paper.title}</h1><p className="md-muted">Enter the password to continue.</p><form onSubmit={handleSubmit} className="login-form"><label className="field-label" htmlFor="password">Password</label><div className="input-with-icon"><LockKeyhole size={17} aria-hidden /><input id="password" className="md-input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" required autoFocus /></div><button type="submit" disabled={loading} className="md-btn md-btn-filled">{loading ? "Signing in..." : "Sign in"}</button>{error && <p className="form-error">{error}</p>}</form></div></main>
  );
}
