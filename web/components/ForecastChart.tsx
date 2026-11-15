"use client";

import { useMemo, useState } from "react";
import { usePrefs } from "@/lib/prefs";
import type { ForecastHour } from "@/lib/types";
import { useWidth } from "./ui";

type Props = { hours: ForecastHour[]; highlight?: number[]; highlightTone?: "accent" | "blue" | "crit"; showCrossCheck?: boolean };

function niceTicks(lo: number, hi: number, count = 5): number[] {
  const span = hi - lo || 1;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * step0).find((s) => span / s <= count) ?? step0 * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

export function ForecastChart({ hours, highlight = [], highlightTone = "accent", showCrossCheck = true }: Props) {
  const { t, n } = usePrefs();
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = width < 520 ? 260 : 340;
  const m = { l: 46, r: 14, t: 18, b: 30 };
  const iw = width - m.l - m.r, ih = H - m.t - m.b;

  const g = useMemo(() => {
    const lo = Math.min(...hours.map((h) => h.p10)), hi = Math.max(...hours.map((h) => h.p90));
    const pad = (hi - lo) * 0.08 || 5;
    const y0 = Math.floor((lo - pad) / 10) * 10, y1 = Math.ceil((hi + pad) / 10) * 10;
    const x = (i: number) => m.l + (hours.length === 1 ? iw / 2 : (i / (hours.length - 1)) * iw);
    const y = (v: number) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
    const line = (key: "forecast" | "cross_check") => hours.map((h, i) => (h[key] == null ? "" : `${i && hours[i - 1][key] != null ? "L" : "M"}${x(i).toFixed(1)},${y(h[key] as number).toFixed(1)}`)).join("");
    const band = hours.map((h, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(h.p90).toFixed(1)}`).join("")
      + hours.map((_, i) => { const j = hours.length - 1 - i; return `L${x(j).toFixed(1)},${y(hours[j].p10).toFixed(1)}`; }).join("") + "Z";
    const iMin = hours.reduce((b, h, i) => (h.forecast < hours[b].forecast ? i : b), 0);
    const iMax = hours.reduce((b, h, i) => (h.forecast > hours[b].forecast ? i : b), 0);
    return { x, y, y0, y1, line, band, iMin, iMax, ticks: niceTicks(y0, y1) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hours, iw, ih]);

  const step = hours.length > 1 ? iw / (hours.length - 1) : iw;
  const tone = highlightTone === "blue" ? "var(--blue)" : highlightTone === "crit" ? "var(--crit)" : "var(--accent)";
  const hv = hover != null ? hours[hover] : null;
  const onMove = (clientX: number, rect: DOMRect) => {
    const i = Math.round(((clientX - rect.left - m.l) / iw) * (hours.length - 1));
    setHover(Math.max(0, Math.min(hours.length - 1, i)));
  };
  const tipLeft = hover != null ? Math.min(Math.max(g.x(hover) - 86, 4), width - 176) : 0;
  const xEvery = width < 520 ? 4 : 2;
  const perHour = hours.some((h) => (h.minute ?? 0) !== 0) ? 4 : 1;
  const clock = (h: ForecastHour) => `${String(h.hour_local).padStart(2, "0")}:${String(h.minute ?? 0).padStart(2, "0")}`;
  const endClock = (h: ForecastHour) => {
    const total = h.hour_local * 60 + (h.minute ?? 0) + 60 / perHour;
    return `${String(Math.floor(total / 60) % 24).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  };

  return (
    <div ref={ref} className="relative select-none">
      <svg width={width} height={H} role="img"
        aria-label={t("Hourly price forecast for tomorrow with a 10 to 90 percent band. Lowest {0} at {1}:00, highest {2} at {3}:00.", `€${n(hours[g.iMin].forecast)}`, hours[g.iMin].hour_local, `€${n(hours[g.iMax].forecast)}`, hours[g.iMax].hour_local)}
        onMouseMove={(e) => onMove(e.clientX, e.currentTarget.getBoundingClientRect())} onMouseLeave={() => setHover(null)}
        onTouchStart={(e) => onMove(e.touches[0].clientX, e.currentTarget.getBoundingClientRect())}
        onTouchMove={(e) => onMove(e.touches[0].clientX, e.currentTarget.getBoundingClientRect())}>
        {g.ticks.map((v) => (
          <g key={v}>
            <line x1={m.l} x2={width - m.r} y1={g.y(v)} y2={g.y(v)} stroke="var(--grid)" strokeWidth="1" />
            <text x={m.l - 8} y={g.y(v) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" className="num">{n(v)}</text>
          </g>
        ))}
        {highlight.map((i) => (
          <rect key={i} x={g.x(i) - step / 2} y={m.t} width={step} height={ih} fill={tone} opacity=".13" rx="4" />
        ))}
        <path d={g.band} fill="var(--series-soft)" />
        {showCrossCheck && <path d={g.line("cross_check")} fill="none" stroke="var(--violet)" strokeWidth="1.5" strokeDasharray="4 4" opacity=".85" />}
        <path d={g.line("forecast")} fill="none" stroke="var(--series)" strokeWidth={perHour === 4 ? 1.75 : 2.25} strokeLinejoin="round" strokeLinecap="round" />
        {[g.iMin, g.iMax].map((i, k) => (
          <g key={k}>
            <circle cx={g.x(i)} cy={g.y(hours[i].forecast)} r="4.5" fill="var(--series)" stroke="var(--surface)" strokeWidth="2" />
            <text x={Math.min(Math.max(g.x(i), m.l + 20), width - m.r - 24)} y={g.y(hours[i].forecast) + (k === 0 ? 20 : -12)} textAnchor="middle" fontSize="12" fontWeight="700" fill="var(--text)" className="num">€{n(hours[i].forecast)}</text>
          </g>
        ))}
        {hours.map((h, i) => ((h.minute ?? 0) === 0 && Math.round(i / perHour) % xEvery === 0 ? (
          <text key={i} x={g.x(i)} y={H - 9} textAnchor="middle" fontSize="11" fill="var(--muted)" className="num">{String(h.hour_local).padStart(2, "0")}</text>
        ) : null))}
        {hv && hover != null && (
          <g pointerEvents="none">
            <line x1={g.x(hover)} x2={g.x(hover)} y1={m.t} y2={m.t + ih} stroke="var(--text-2)" strokeWidth="1" strokeDasharray="3 3" />
            <circle cx={g.x(hover)} cy={g.y(hv.forecast)} r="5" fill="var(--series)" stroke="var(--surface)" strokeWidth="2" />
          </g>
        )}
      </svg>
      {hv && (
        <div className="pointer-events-none absolute top-2 w-[172px] rounded-xl border border-border bg-surface-2 p-3 text-sm shadow-lg" style={{ left: tipLeft }}>
          <div className="num font-bold">{clock(hv)} – {endClock(hv)}</div>
          <div className="mt-1.5 flex justify-between gap-3"><span className="text-muted">{t("Forecast")}</span><span className="num font-bold">€{n(hv.forecast, 1)}</span></div>
          <div className="flex justify-between gap-3"><span className="text-muted">{t("Likely range")}</span><span className="num">€{n(hv.p10)}–{n(hv.p90)}</span></div>
          {hv.spike_probability != null && <div className="flex justify-between gap-3"><span className="text-muted">{t("Spike risk")}</span><span className="num">{n(hv.spike_probability * 100)}%</span></div>}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[.8rem] text-text-2">
        <span className="inline-flex items-center gap-2"><span aria-hidden className="inline-block h-0.5 w-5 rounded" style={{ background: "var(--series)" }} />{t("Forecast (EUR/MWh)")}</span>
        <span className="inline-flex items-center gap-2"><span aria-hidden className="inline-block h-3 w-5 rounded-sm" style={{ background: "var(--series-soft)" }} />{t("10–90 % likely range")}</span>
        {showCrossCheck && hours.some((h) => h.cross_check != null) && (
          <span className="inline-flex items-center gap-2"><span aria-hidden className="inline-block w-5 border-t-2 border-dashed" style={{ borderColor: "var(--violet)" }} />{t("Cross-check model")}</span>
        )}
      </div>
    </div>
  );
}
