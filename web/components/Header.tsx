"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { usePrefs } from "@/lib/prefs";
import type { Status } from "@/lib/types";

const NAV = [
  { href: "/", label: "Tomorrow" },
  { href: "/battery", label: "Battery" },
  { href: "/track-record", label: "Track record" },
  { href: "/market", label: "Market" },
  { href: "/model", label: "Model" },
];

function Bolt() {
  return (
    <span aria-hidden className="grid h-8 w-8 place-items-center rounded-[10px] text-[#1A1203]"
      style={{ background: "linear-gradient(135deg,#F5A524,#E0701B)", boxShadow: "0 4px 14px rgba(245,165,36,.28)" }}>
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M13.2 2 4.5 13.4h6L9.6 22l9-11.6h-6z" /></svg>
    </span>
  );
}

export function Header({ status }: { status: Status | null }) {
  const path = usePathname();
  const { t, lang, setLang, theme, setTheme } = usePrefs();
  const [open, setOpen] = useState(false);
  const fresh = status?.ok;
  const isCurrent = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));

  return (
    <header className="sticky top-0 z-40 border-b border-border" style={{ background: "color-mix(in srgb, var(--bg) 82%, transparent)", backdropFilter: "blur(14px) saturate(1.2)" }}>
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-3 px-4 sm:px-6">
        <Link href="/" className="flex shrink-0 items-center gap-2.5 whitespace-nowrap" aria-label="BG Power, home">
          <Bolt />
          <span className="leading-none">
            <span className="block text-[15px] font-extrabold tracking-tight">BG Power</span>
            <span className="block text-[10px] font-semibold uppercase tracking-[.12em] text-muted">SDAC · EET</span>
          </span>
        </Link>

        <nav aria-label={t("Pages")} className="ml-3 hidden items-center gap-1 md:flex">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} aria-current={isCurrent(n.href) ? "page" : undefined}
              className="pill inline-flex items-center border-transparent">{t(n.label)}</Link>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          {status && (
            <span className="hidden lg:block">
              <span className="chip" title={status.problems.join("; ") || t("Forecast is up to date")} role="status">
                <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: fresh ? "var(--good)" : "var(--warn)" }} />
                {fresh ? t("Live") : t("Check data")}
              </span>
            </span>
          )}
          <div className="flex overflow-hidden rounded-full border border-border" role="group" aria-label={t("Language")}>
            {(["en", "bg"] as const).map((l) => (
              <button key={l} onClick={() => setLang(l)} aria-pressed={lang === l}
                className="h-9 px-3 text-xs font-bold uppercase transition-colors"
                style={lang === l ? { background: "color-mix(in srgb, var(--accent) 18%, transparent)", color: "var(--text)" } : { color: "var(--muted)" }}>{l}</button>
            ))}
          </div>
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} className="grid h-9 w-9 place-items-center rounded-full border border-border text-text-2 transition-colors hover:border-accent hover:text-text"
            aria-label={theme === "dark" ? t("Switch to light theme") : t("Switch to dark theme")}>
            {theme === "dark"
              ? <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
              : <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>}
          </button>
          <button onClick={() => setOpen((o) => !o)} className="grid h-9 w-9 place-items-center rounded-full border border-border text-text-2 md:hidden"
            aria-expanded={open} aria-controls="mobile-nav" aria-label={t("Menu")}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">{open ? <path d="M6 6l12 12M18 6 6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}</svg>
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label={t("Pages")} className="fade-in border-t border-border px-4 py-3 md:hidden">
          <ul className="grid gap-1">
            {NAV.map((n) => (
              <li key={n.href}>
                <Link href={n.href} aria-current={isCurrent(n.href) ? "page" : undefined} onClick={() => setOpen(false)}
                  className="flex min-h-11 items-center rounded-xl px-3 font-semibold text-text-2 aria-[current=page]:bg-surface-2 aria-[current=page]:text-text">{t(n.label)}</Link>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
}
