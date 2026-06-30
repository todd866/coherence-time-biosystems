
"use client";

import { FileDown, LogOut, Moon, Sun } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { navLinks, paper } from "@/data/site";

export default function Header() {
  const pathname = usePathname();
  const router = useRouter();
  const [dark, setDark] = useState(false);
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); setDark(document.documentElement.classList.contains("dark")); }, []);
  function toggleTheme() {
    const next = !dark; setDark(next); document.documentElement.classList.toggle("dark", next);
    try { localStorage.setItem("theme", next ? "dark" : "light"); } catch {}
  }
  async function handleLogout() {
    await fetch("/api/auth", { method: "DELETE" });
    router.push("/login");
  }
  if (pathname === "/login") return null;
  return (
    <header className="site-header">
      <a href="/" aria-label="Home" title="Home" className="site-mark"><span aria-hidden>{paper.mark}</span></a>
      <nav className="header-nav" aria-label="Primary">{navLinks.map((link) => { const Icon = link.icon; const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href); return <a key={link.href} href={link.href} className={`md-btn md-btn-text header-link${active ? " active" : ""}`} aria-current={active ? "page" : undefined}><Icon size={16} aria-hidden /><span>{link.label}</span></a>; })}</nav>
      <span className="header-spacer" />
      <a href="/paper.pdf" className="md-btn md-btn-tonal header-action"><FileDown size={16} aria-hidden /><span>PDF</span></a>
      <button onClick={toggleTheme} className="icon-button" aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} title={dark ? "Light mode" : "Dark mode"}>{mounted ? (dark ? <Sun size={17} aria-hidden /> : <Moon size={17} aria-hidden />) : null}</button>
      <button onClick={handleLogout} className="icon-button" aria-label="Sign out" title="Sign out"><LogOut size={17} aria-hidden /></button>
    </header>
  );
}
