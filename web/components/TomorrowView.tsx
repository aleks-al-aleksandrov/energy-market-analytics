"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { planDay } from "@/lib/battery";
import { usePrefs } from "@/lib/prefs";
import type { Forecast, TrackRecord } from "@/lib/types";
import { ForecastChart } from "./ForecastChart";
import { Empty, Notice, PageHead, Stat, TableToggle } from "./ui";

type Q = "cheap" | "avoid" | "spike" | "battery" | "trust";
const hh = (h: number) => `${String(h).padStart(2, "0")}:00`;

function window3(hours: Forecast["hours"], pick: "min" | "max"): number[] {
  let best = 0, bestV = pick === "min" ? Infinity : -Infinity;
  for (let i = 0; i + 2 < hours.length; i++) {
    const v = hours[i].forecast + hours[i + 1].forecast + hours[i + 2].forecast;
    if (pick === "min" ? v < bestV : v > bestV) { best = i; bestV = v; }
  }
  return [best, best + 1, best + 2];
}

export function TomorrowView({ forecast, track }: { forecast: Forecast | null; track: TrackRecord | null }) {
  const { t, n, date } = usePrefs();
  const [q, setQ] = useState<Q>("cheap");
  const [res, setRes] = useState<"hour" | "quarter">("hour");

  const d = useMemo(() => {
    if (!forecast?.available) return null;
    const hours = forecast.hours;
    const cheap = window3(hours, "min"), dear = window3(hours, "max");
    const mean = (ix: number[]) => ix.reduce((s, i) => s + hours[i].forecast, 0) / ix.length;
    const spike = hours.map((h, i) => ((h.spike_probability ?? 0) >= 0.1 ? i : -1)).filter((i) => i >= 0);
    const plan = planDay(hours.map((h) => h.forecast), { powerMw: 1, durationH: 2, efficiency: 0.9, cyclesPerDay: 1, degradationEurMwh: 0 });
    const charge = plan.net.map((x, i) => (x < -1e-6 ? i : -1)).filter((i) => i >= 0);
    const discharge = plan.net.map((x, i) => (x > 1e-6 ? i : -1)).filter((i) => i >= 0);
    return { hours, cheap, dear, cheapMean: mean(cheap), dearMean: mean(dear), spike, plan, charge, discharge };
  }, [forecast]);

  if (!forecast?.available || !d) {
    return <Empty title={t("No forecast yet")} body={t("The daily run has not published a forecast. It runs every morning before the 12:00 CET auction.")} />;
  }
  const s = forecast.summary;
  const totals = track?.totals;
  const skill = totals?.skill_vs_naive ?? null;
  const range = (ix: number[]) => `${hh(d.hours[ix[0]].hour_local)}–${hh((d.hours[ix[ix.length - 1]].hour_local + 1) % 24)}`;

  const answers: Record<Q, { label: string; text: string; highlight: number[]; tone: "accent" | "blue" | "crit" }> = {
    cheap: {
      label: t("When is power cheapest?"), highlight: d.cheap, tone: "blue",
      text: t("The cheapest three hours are {0}, averaging €{1}/MWh, {2}% below the day's average. If you can shift an EV charge, a heat pump or a wash, this is the window.", range(d.cheap), n(d.cheapMean), n((1 - d.cheapMean / s.mean) * 100)),
    },
    avoid: {
      label: t("When should I avoid it?"), highlight: d.dear, tone: "crit",
      text: t("The dearest three hours are {0}, averaging €{1}/MWh, {2}% above the day's average.", range(d.dear), n(d.dearMean), n((d.dearMean / s.mean - 1) * 100)),
    },
    spike: {
      label: t("Is a price spike likely?"), highlight: d.spike, tone: "crit",
      text: d.spike.length
        ? t("{0} hours carry a spike risk of 10% or more; the highest is {1}%. A spike means the price jumps well above yesterday's average.", d.spike.length, n((s.max_spike_probability ?? 0) * 100))
        : t("No hour reaches a 10% spike risk; the highest is {0}%.", n((s.max_spike_probability ?? 0) * 100)),
    },
    battery: {
      label: t("What would a battery earn?"), highlight: [...d.charge, ...d.discharge], tone: "accent",
      text: t("A 1 MW / 2 MWh battery would charge around {0} and sell around {1}, for about €{2} if prices match the forecast. Realised revenue is usually lower.", hh(d.hours[d.charge[0] ?? 0].hour_local), hh(d.hours[d.discharge[0] ?? 0].hour_local), n(d.plan.revenue)),
    },
    trust: {
      label: t("Can I trust this?"), highlight: [], tone: "accent",
      text: totals && track?.n_days
        ? t("Over {0} live days the forecast missed by €{1}/MWh on average, against €{2} for 'same as yesterday'. The band held the price {3}% of the time (target 80%). {0} days is too few to prove anything.", track.n_days, n(totals.mae_model), n(totals.mae_naive), n((totals.coverage_10_90 ?? 0) * 100))
        : t("There is no live record yet. In back-tests the forecast beat 'same as yesterday' by about 21%, but it struggles when prices jump to new levels."),
    },
  };
  const a = answers[q];
  const quarters = forecast.quarters?.length ? forecast.quarters.map((x) => ({ ...x, spike_probability: null, cross_check: null })) : null;
  const fine = res === "quarter" && quarters != null;
  const points = fine ? quarters! : d.hours;
  // a highlighted hour covers its four quarters
  const lit = fine ? a.highlight.flatMap((i) => quarters!.map((x, j) => (x.hour_local === d.hours[i].hour_local ? j : -1)).filter((j) => j >= 0)) : a.highlight;

  return (
    <>
      <PageHead eyebrow={t("Tomorrow · {0}", date(forecast.delivery_day))}
        title={t("Around €{0}/MWh, cheapest near {1}, peaking near {2}", n(s.mean), hh(s.min_hour_local), hh(s.max_hour_local))}
        lede={t("Hourly day-ahead price forecast for Bulgaria, made before the 12:00 CET auction. The shaded range is where the price should land 8 times out of 10.")} />

      <div className="mb-4 flex flex-wrap gap-2">
        <span className="chip">{t("Published {0}", forecast.generated_at ? date(forecast.generated_at, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "–")}</span>
        <span className="chip">{t("Model trained to {0}", forecast.model_trained_until ?? "–")}</span>
        <span className="chip">{t("Band adjusted to {0}", forecast.band.calibrated_as_of ?? "–")}</span>
      </div>

      {forecast.stale && <div className="mb-4"><Notice tone="warn">{t("This forecast is {0} hours old. The daily run may have failed; treat the numbers with care.", n(forecast.age_hours))}</Notice></div>}
      {s.models_disagree && <div className="mb-4"><Notice tone="warn">{t("Two models disagree on tomorrow's level by {0}%. That usually means prices are moving into a new regime, where this forecast is less reliable.", n((s.models_disagree_by ?? 0) * 100))}</Notice></div>}

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label={t("Key numbers")}>
        <Stat label={t("Average")} value={`€${n(s.mean)}`} sub={t("EUR/MWh, base load")} />
        <Stat label={t("Lowest hour")} value={`€${n(s.min)}`} sub={hh(s.min_hour_local)} />
        <Stat label={t("Highest hour")} value={`€${n(s.max)}`} sub={hh(s.max_hour_local)} />
        <Stat label={t("Daily spread")} value={`€${n(s.spread)}`} sub={t("what flexibility can capture")} />
      </section>

      <section className="card card-pad mt-4 fade-in" aria-label={t("Forecast chart")}>
        <h2 className="sr-only">{t("Forecast chart")}</h2>
        <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label={t("Ask the forecast")}>
          {(Object.keys(answers) as Q[]).map((k) => (
            <button key={k} className="pill" aria-pressed={q === k} onClick={() => setQ(k)}>{answers[k].label}</button>
          ))}
        </div>
        <p className="mb-4 max-w-3xl rounded-xl border border-border bg-surface-2 p-4 leading-relaxed" aria-live="polite">{a.text}</p>
        {quarters && (
          <div className="mb-3 flex flex-wrap items-center gap-2" role="group" aria-label={t("Resolution")}>
            <button className="pill" aria-pressed={!fine} onClick={() => setRes("hour")}>{t("Hourly")}</button>
            <button className="pill" aria-pressed={fine} onClick={() => setRes("quarter")}>{t("15 minutes")}</button>
            <span className="text-sm text-muted">{t("The market clears in 15-minute steps; the quarter view adds the shape inside each hour.")}</span>
          </div>
        )}
        <ForecastChart hours={points} highlight={lit} highlightTone={a.tone} showCrossCheck={!fine} />
        <TableToggle>
          <table className="data">
            <thead><tr><th>{fine ? t("Time") : t("Hour")}</th><th className="n">{t("Forecast")}</th><th className="n">{t("Low (10%)")}</th><th className="n">{t("High (90%)")}</th><th className="n">{t("Spike risk")}</th></tr></thead>
            <tbody>
              {points.map((h) => (
                <tr key={h.timestamp_utc}><td className="num">{`${String(h.hour_local).padStart(2, "0")}:${String(h.minute ?? 0).padStart(2, "0")}`}</td><td className="n font-semibold">{n(h.forecast, 1)}</td><td className="n">{n(h.p10, 1)}</td><td className="n">{n(h.p90, 1)}</td><td className="n">{h.spike_probability == null ? "–" : `${n(h.spike_probability * 100)}%`}</td></tr>
              ))}
            </tbody>
          </table>
        </TableToggle>
      </section>

      <section className="mt-4 grid gap-3 md:grid-cols-3">
        <Link href="/battery" className="card card-pad transition-transform hover:-translate-y-0.5">
          <div className="eyebrow">{t("Battery")}</div>
          <h2 className="mt-1 text-lg font-bold">{t("Plan a battery for tomorrow")}</h2>
          <p className="mt-1 text-sm text-text-2">{t("Set size and efficiency, see when to charge and sell, and what it could earn.")}</p>
        </Link>
        <Link href="/track-record" className="card card-pad transition-transform hover:-translate-y-0.5">
          <div className="eyebrow">{t("Track record")}</div>
          <h2 className="mt-1 text-lg font-bold">{skill == null ? t("How it has done live") : t("{0}% better than 'same as yesterday', live", n(skill * 100))}</h2>
          <p className="mt-1 text-sm text-text-2">{t("Every forecast is logged before the auction and scored afterwards. Misses are shown too.")}</p>
        </Link>
        <Link href="/model" className="card card-pad transition-transform hover:-translate-y-0.5">
          <div className="eyebrow">{t("Model")}</div>
          <h2 className="mt-1 text-lg font-bold">{t("How it works and where it fails")}</h2>
          <p className="mt-1 text-sm text-text-2">{t("Method, back-test results and the changes that were tested and rejected.")}</p>
        </Link>
      </section>
    </>
  );
}
