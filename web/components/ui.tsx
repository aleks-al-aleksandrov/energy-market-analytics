"use client";

import { useEffect, useRef, useState } from "react";
import { usePrefs } from "@/lib/prefs";

export function PageHead({ eyebrow, title, lede }: { eyebrow: string; title: string; lede?: string }) {
  return (
    <div className="fade-in mb-6 sm:mb-8">
      <div className="eyebrow">{eyebrow}</div>
      <h1 className="mt-1 text-[clamp(1.7rem,4.2vw,2.6rem)] font-extrabold leading-[1.08] tracking-[-.035em]">{title}</h1>
      {lede && <p className="mt-3 max-w-3xl text-[1.02rem] leading-relaxed text-text-2">{lede}</p>}
    </div>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "good" | "warn" | "crit" }) {
  const color = tone ? `var(--${tone})` : "var(--text)";
  return (
    <div className="card card-pad">
      <div className="text-[.72rem] font-semibold uppercase tracking-[.07em] text-muted">{label}</div>
      <div className="num mt-1.5 text-[clamp(1.4rem,3.4vw,1.9rem)] font-bold tracking-tight" style={{ color }}>{value}</div>
      {sub && <div className="mt-1 text-sm leading-snug text-muted">{sub}</div>}
    </div>
  );
}

export function Notice({ tone = "warn", children }: { tone?: "warn" | "crit" | "good"; children: React.ReactNode }) {
  return (
    <div role="note" className="flex gap-3 rounded-2xl border p-4 text-sm leading-relaxed"
      style={{ borderColor: `color-mix(in srgb, var(--${tone}) 45%, transparent)`, background: `color-mix(in srgb, var(--${tone}) 9%, transparent)` }}>
      <span aria-hidden className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: `var(--${tone})` }} />
      <div>{children}</div>
    </div>
  );
}

export function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="card card-pad text-center">
      <div className="text-lg font-bold">{title}</div>
      <p className="mx-auto mt-2 max-w-md text-text-2">{body}</p>
    </div>
  );
}

/** Width of a container, for responsive SVG charts. */
export function useWidth<T extends HTMLElement>(initial = 720): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((e) => setW(Math.max(280, Math.round(e[0].contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Every chart offers its numbers as a table. */
export function TableToggle({ children }: { children: React.ReactNode }) {
  const { t } = usePrefs();
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button className="pill inline-flex items-center gap-2" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <svg aria-hidden viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 5h18v14H3zM3 10h18M9 5v14" /></svg>
        {open ? t("Hide table") : t("Table view")}
      </button>
      {open && <div className="fade-in mt-3 overflow-x-auto rounded-xl border border-border">{children}</div>}
    </div>
  );
}

export const eur = (n: (v: number | null | undefined, d?: number) => string, v: number | null | undefined, d = 0) => `€${n(v, d)}`;
