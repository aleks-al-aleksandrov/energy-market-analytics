"use client";

import { useState } from "react";
import { usePrefs } from "@/lib/prefs";
import { useWidth } from "./ui";

function ticks(lo: number, hi: number, count = 5): number[] {
  const span = hi - lo || 1;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * step0).find((s) => span / s <= count) ?? step0 * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

export type Series = { key: string; label: string; color: string; values: (number | null)[]; dashed?: boolean; emphasis?: boolean };

/** Multi-series line chart with a shared crosshair tooltip and a legend. One y-axis only. */
export function LineChart({ labels, series, unit = "€", ariaLabel, every }: { labels: string[]; series: Series[]; unit?: string; ariaLabel: string; every?: number }) {
  const { n } = usePrefs();
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = width < 520 ? 250 : 320, m = { l: 46, r: 14, t: 14, b: 30 };
  const iw = width - m.l - m.r, ih = H - m.t - m.b;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v != null));
  const lo0 = Math.min(...all), hi0 = Math.max(...all), pad = (hi0 - lo0) * 0.08 || 1;
  const tk = ticks(lo0 - pad, hi0 + pad);
  const y0 = Math.min(lo0 - pad, tk[0] ?? lo0), y1 = Math.max(hi0 + pad, tk[tk.length - 1] ?? hi0);
  const x = (i: number) => m.l + (labels.length < 2 ? iw / 2 : (i / (labels.length - 1)) * iw);
  const y = (v: number) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
  const path = (vals: (number | null)[]) => vals.map((v, i) => (v == null ? "" : `${i && vals[i - 1] != null ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)).join("");
  const step = every ?? Math.max(1, Math.ceil(labels.length / (width < 520 ? 5 : 9)));
  const move = (cx: number, r: DOMRect) => setHover(Math.max(0, Math.min(labels.length - 1, Math.round(((cx - r.left - m.l) / iw) * (labels.length - 1)))));
  const tipLeft = hover != null ? Math.min(Math.max(x(hover) - 90, 4), width - 190) : 0;
  return (
    <div ref={ref} className="relative select-none">
      <svg width={width} height={H} role="img" aria-label={ariaLabel}
        onMouseMove={(e) => move(e.clientX, e.currentTarget.getBoundingClientRect())} onMouseLeave={() => setHover(null)}
        onTouchStart={(e) => move(e.touches[0].clientX, e.currentTarget.getBoundingClientRect())} onTouchMove={(e) => move(e.touches[0].clientX, e.currentTarget.getBoundingClientRect())}>
        {tk.map((v) => (<g key={v}><line x1={m.l} x2={width - m.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" /><text x={m.l - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" className="num">{n(v)}</text></g>))}
        {series.map((s) => (<path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={s.emphasis ? 2.5 : 1.75} strokeDasharray={s.dashed ? "4 4" : undefined} strokeLinejoin="round" strokeLinecap="round" opacity={s.emphasis ? 1 : 0.9} />))}
        {labels.map((l, i) => (i % step === 0 ? <text key={i} x={x(i)} y={H - 9} textAnchor="middle" fontSize="11" fill="var(--muted)">{l}</text> : null))}
        {hover != null && (<g pointerEvents="none"><line x1={x(hover)} x2={x(hover)} y1={m.t} y2={m.t + ih} stroke="var(--text-2)" strokeDasharray="3 3" />
          {series.map((s) => (s.values[hover] == null ? null : <circle key={s.key} cx={x(hover)} cy={y(s.values[hover] as number)} r="4.5" fill={s.color} stroke="var(--surface)" strokeWidth="2" />))}</g>)}
      </svg>
      {hover != null && (
        <div className="pointer-events-none absolute top-2 w-[186px] rounded-xl border border-border bg-surface-2 p-3 text-sm shadow-lg" style={{ left: tipLeft }}>
          <div className="font-bold">{labels[hover]}</div>
          {series.map((s) => (<div key={s.key} className="mt-1 flex items-center justify-between gap-3"><span className="inline-flex items-center gap-2 text-muted"><span aria-hidden className="h-2 w-2 rounded-full" style={{ background: s.color }} />{s.label}</span><span className="num font-semibold">{s.values[hover] == null ? "–" : `${unit}${n(s.values[hover], 0)}`}</span></div>))}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[.8rem] text-text-2">
        {series.map((s) => (<span key={s.key} className="inline-flex items-center gap-2"><span aria-hidden className="inline-block w-5 border-t-2" style={{ borderColor: s.color, borderStyle: s.dashed ? "dashed" : "solid" }} />{s.label}</span>))}
      </div>
    </div>
  );
}

/** Two bars per category (for example forecast error vs naive error per day). */
export function PairBars({ labels, a, b, ariaLabel }: { labels: string[]; a: { label: string; color: string; values: number[] }; b: { label: string; color: string; values: number[] }; ariaLabel: string }) {
  const { n } = usePrefs();
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = 280, m = { l: 46, r: 10, t: 14, b: 30 };
  const iw = width - m.l - m.r, ih = H - m.t - m.b;
  const hi = Math.max(...a.values, ...b.values, 1) * 1.08;
  const tk = ticks(0, hi);
  const y = (v: number) => m.t + ih - (v / hi) * ih;
  const band = iw / Math.max(labels.length, 1), bw = Math.max(3, Math.min(18, band * 0.32));
  const step = Math.max(1, Math.ceil(labels.length / (width < 520 ? 5 : 10)));
  return (
    <div ref={ref} className="relative select-none">
      <svg width={width} height={H} role="img" aria-label={ariaLabel} onMouseLeave={() => setHover(null)}>
        {tk.map((v) => (<g key={v}><line x1={m.l} x2={width - m.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" /><text x={m.l - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" className="num">{n(v)}</text></g>))}
        {labels.map((l, i) => {
          const cx = m.l + band * (i + 0.5);
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onTouchStart={() => setHover(i)}>
              <rect x={m.l + band * i} y={m.t} width={band} height={ih} fill={hover === i ? "var(--surface-2)" : "transparent"} />
              <rect x={cx - bw - 1} y={y(a.values[i])} width={bw} height={Math.max(0, m.t + ih - y(a.values[i]))} rx="3" fill={a.color} />
              <rect x={cx + 1} y={y(b.values[i])} width={bw} height={Math.max(0, m.t + ih - y(b.values[i]))} rx="3" fill={b.color} />
              {i % step === 0 && <text x={cx} y={H - 9} textAnchor="middle" fontSize="11" fill="var(--muted)">{l}</text>}
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <div className="pointer-events-none absolute top-2 w-[190px] rounded-xl border border-border bg-surface-2 p-3 text-sm shadow-lg" style={{ left: Math.min(Math.max(m.l + band * (hover + 0.5) - 95, 4), width - 194) }}>
          <div className="font-bold">{labels[hover]}</div>
          {[a, b].map((s) => (<div key={s.label} className="mt-1 flex items-center justify-between gap-3"><span className="inline-flex items-center gap-2 text-muted"><span aria-hidden className="h-2 w-2 rounded-full" style={{ background: s.color }} />{s.label}</span><span className="num font-semibold">€{n(s.values[hover], 1)}</span></div>))}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[.8rem] text-text-2">
        {[a, b].map((s) => (<span key={s.label} className="inline-flex items-center gap-2"><span aria-hidden className="inline-block h-3 w-3 rounded-sm" style={{ background: s.color }} />{s.label}</span>))}
      </div>
    </div>
  );
}

/** Battery schedule: charge / discharge bars per hour over the price line, with the state of charge underneath. */
export function ScheduleChart({ hours, prices, net, soc, energy }: { hours: number[]; prices: number[]; net: number[]; soc: number[]; energy: number }) {
  const { t, n } = usePrefs();
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = 330, m = { l: 46, r: 12, t: 14, b: 30 }, socH = 54, gap = 18;
  const iw = width - m.l - m.r, ih = H - m.t - m.b - socH - gap;
  const lo = Math.min(...prices), hi = Math.max(...prices), pad = (hi - lo) * 0.1 || 5;
  const tk = ticks(lo - pad, hi + pad, 4), y0 = Math.min(lo - pad, tk[0]), y1 = Math.max(hi + pad, tk[tk.length - 1]);
  const band = iw / hours.length;
  const x = (i: number) => m.l + band * (i + 0.5);
  const y = (v: number) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
  const maxP = Math.max(...net.map(Math.abs), 1e-9);
  const socTop = m.t + ih + gap;
  const line = prices.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p).toFixed(1)}`).join("");
  const socPath = `M${x(0) - band / 2},${socTop + socH}` + soc.map((s, i) => `L${x(i) + band / 2},${(socTop + socH - (s / Math.max(energy, 1e-9)) * socH).toFixed(1)}`).join("") + `L${x(soc.length - 1) + band / 2},${socTop + socH}Z`;
  return (
    <div ref={ref} className="relative select-none">
      <svg width={width} height={H} role="img" aria-label={t("Battery schedule by hour: charging in cheap hours, selling in expensive hours, with the state of charge below.")} onMouseLeave={() => setHover(null)}>
        {tk.map((v) => (<g key={v}><line x1={m.l} x2={width - m.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" /><text x={m.l - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" className="num">{n(v)}</text></g>))}
        {net.map((v, i) => {
          const active = Math.abs(v) > 1e-6;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onTouchStart={() => setHover(i)}>
              <rect x={m.l + band * i} y={m.t} width={band} height={ih + gap + socH} fill={hover === i ? "var(--surface-2)" : "transparent"} />
              {active && <rect x={m.l + band * i + 2} y={m.t} width={Math.max(2, band - 4)} height={ih} rx="5" fill={v < 0 ? "var(--blue)" : "var(--accent)"} opacity={0.14 + 0.2 * (Math.abs(v) / maxP)} />}
              {active && <text x={x(i)} y={m.t + 14} textAnchor="middle" fontSize="11" fontWeight="700" fill={v < 0 ? "var(--blue)" : "var(--accent)"}>{v < 0 ? "↓" : "↑"}</text>}
            </g>
          );
        })}
        <path d={line} fill="none" stroke="var(--series)" strokeWidth="2.25" strokeLinejoin="round" pointerEvents="none" />
        <path d={socPath} fill="var(--series-soft)" stroke="var(--series)" strokeWidth="1.25" pointerEvents="none" />
        <text x={m.l - 8} y={socTop + 10} textAnchor="end" fontSize="10" fill="var(--muted)">{t("Stored")}</text>
        {hours.map((h, i) => (i % (width < 520 ? 4 : 2) === 0 ? <text key={i} x={x(i)} y={H - 9} textAnchor="middle" fontSize="11" fill="var(--muted)" className="num">{String(h).padStart(2, "0")}</text> : null))}
      </svg>
      {hover != null && (
        <div className="pointer-events-none absolute top-2 w-[190px] rounded-xl border border-border bg-surface-2 p-3 text-sm shadow-lg" style={{ left: Math.min(Math.max(x(hover) - 95, 4), width - 194) }}>
          <div className="num font-bold">{String(hours[hover]).padStart(2, "0")}:00</div>
          <div className="mt-1 flex justify-between gap-3"><span className="text-muted">{t("Price")}</span><span className="num font-semibold">€{n(prices[hover], 1)}</span></div>
          <div className="flex justify-between gap-3"><span className="text-muted">{net[hover] < -1e-6 ? t("Charge") : net[hover] > 1e-6 ? t("Sell") : t("Idle")}</span><span className="num">{n(Math.abs(net[hover]), 2)} MW</span></div>
          <div className="flex justify-between gap-3"><span className="text-muted">{t("Stored")}</span><span className="num">{n(soc[hover], 2)} MWh</span></div>
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[.8rem] text-text-2">
        <span className="inline-flex items-center gap-2"><span aria-hidden className="inline-block h-0.5 w-5 rounded" style={{ background: "var(--series)" }} />{t("Forecast price")}</span>
        <span className="inline-flex items-center gap-2"><span aria-hidden className="font-bold" style={{ color: "var(--blue)" }}>↓</span>{t("Charge")}</span>
        <span className="inline-flex items-center gap-2"><span aria-hidden className="font-bold" style={{ color: "var(--accent)" }}>↑</span>{t("Sell")}</span>
        <span className="inline-flex items-center gap-2"><span aria-hidden className="inline-block h-3 w-5 rounded-sm" style={{ background: "var(--series-soft)" }} />{t("Energy stored")}</span>
      </div>
    </div>
  );
}
