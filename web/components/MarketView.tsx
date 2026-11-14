"use client";

import { useMemo, useState } from "react";
import { usePrefs } from "@/lib/prefs";
import type { Market } from "@/lib/types";
import { LineChart, type Series } from "./charts";
import { Empty, PageHead, Stat, TableToggle } from "./ui";

const ZONE_COLOR: Record<string, string> = { BG: "var(--series)", RO: "var(--blue)", GR: "var(--magenta)", HU: "var(--green)" };
const ZONE_NAME: Record<string, string> = { BG: "Bulgaria", RO: "Romania", GR: "Greece", HU: "Hungary" };

export function MarketView({ market }: { market: Market | null }) {
  const { t, n, date } = usePrefs();
  const [span, setSpan] = useState<7 | 30>(30);
  const d = useMemo(() => {
    if (!market?.available || !market.daily?.length) return null;
    const daily = market.daily.slice(-span);
    const zones = market.zones ?? ["BG"];
    const num = (v: unknown) => (typeof v === "number" ? v : null);
    const series: Series[] = zones.map((z) => ({ key: z, label: z, color: ZONE_COLOR[z] ?? "var(--neutral)", values: daily.map((r) => num(r[`${z}_mean`])), emphasis: z === "BG" }));
    const bg = daily.map((r) => num(r.BG_mean)).filter((v): v is number => v != null);
    const mean = bg.reduce((s, v) => s + v, 0) / Math.max(bg.length, 1);
    const lows = daily.map((r) => num(r.BG_min)).filter((v): v is number => v != null);
    const highs = daily.map((r) => num(r.BG_max)).filter((v): v is number => v != null);
    const spreadVsRo = daily.map((r) => (num(r.BG_mean) != null && num(r.RO_mean) != null ? (num(r.BG_mean) as number) - (num(r.RO_mean) as number) : null)).filter((v): v is number => v != null);
    return { daily, zones, series, mean, min: Math.min(...lows), max: Math.max(...highs), last: bg[bg.length - 1], first: bg[0],
      vsRo: spreadVsRo.reduce((s, v) => s + v, 0) / Math.max(spreadVsRo.length, 1) };
  }, [market, span]);

  if (!d || !market) return (<><PageHead eyebrow={t("Market")} title={t("No market data yet")} /><Empty title={t("Nothing to show")} body={t("Published prices appear here after the first daily run.")} /></>);
  const change = d.first ? (d.last / d.first - 1) * 100 : 0;
  return (
    <>
      <PageHead eyebrow={t("Market · last {0} days", span)}
        title={t("Bulgaria averaged €{0}/MWh, between €{1} and €{2}", n(d.mean), n(d.min), n(d.max))}
        lede={t("Published day-ahead prices for Bulgaria and its coupled neighbours. When the interconnectors have spare capacity the zones clear at the same price; when they congest, prices split.")} />
      <div className="mb-4 flex flex-wrap items-center gap-2" role="group" aria-label={t("Period")}>
        {([7, 30] as const).map((s) => (<button key={s} className="pill" aria-pressed={span === s} onClick={() => setSpan(s)}>{t("{0} days", s)}</button>))}
        <span className="chip ml-auto">{t("Data through {0}", market.data_through ? date(market.data_through, { day: "numeric", month: "short", year: "numeric" }) : "–")}</span>
      </div>
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label={t("Key numbers")}>
        <Stat label={t("Average price")} value={`€${n(d.mean)}`} sub={t("Bulgaria, daily base load")} />
        <Stat label={t("Latest day")} value={`€${n(d.last)}`} sub={t("{0}% vs start of period", `${change >= 0 ? "+" : ""}${n(change)}`)} />
        <Stat label={t("Hourly range")} value={`€${n(d.min)} – ${n(d.max)}`} sub={t("lowest and highest hour")} />
        <Stat label={t("Bulgaria vs Romania")} value={`${d.vsRo >= 0 ? "+" : "−"}€${n(Math.abs(d.vsRo), 1)}`} sub={t("average daily gap")} />
      </section>
      <section className="card card-pad mt-4" aria-label={t("Daily average price by zone")}>
        <h2 className="mb-4 text-lg font-bold">{t("Daily average price by zone")}</h2>
        <LineChart labels={d.daily.map((r) => date(r.day, { day: "numeric", month: "short" }))} series={d.series.map((s) => ({ ...s, label: t(ZONE_NAME[s.key] ?? s.key) }))}
          ariaLabel={t("Daily average day-ahead price for Bulgaria, Romania, Greece and Hungary.")} />
        <TableToggle>
          <table className="data">
            <thead><tr><th>{t("Day")}</th>{d.zones.map((z) => <th key={z} className="n">{z}</th>)}<th className="n">BG {t("low")}</th><th className="n">BG {t("high")}</th></tr></thead>
            <tbody>
              {[...d.daily].reverse().map((r) => (
                <tr key={r.day}><td>{date(r.day, { weekday: "short", day: "numeric", month: "short" })}</td>
                  {d.zones.map((z) => <td key={z} className="n">{n(r[`${z}_mean`] as number | null, 1)}</td>)}
                  <td className="n">{n(r.BG_min as number | null, 1)}</td><td className="n">{n(r.BG_max as number | null, 1)}</td></tr>
              ))}
            </tbody>
          </table>
        </TableToggle>
      </section>
    </>
  );
}
