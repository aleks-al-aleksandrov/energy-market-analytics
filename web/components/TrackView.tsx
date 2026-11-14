"use client";

import { usePrefs } from "@/lib/prefs";
import type { TrackRecord } from "@/lib/types";
import { PairBars } from "./charts";
import { Empty, Notice, PageHead, Stat, TableToggle } from "./ui";

export function TrackView({ track }: { track: TrackRecord | null }) {
  const { t, n, date } = usePrefs();
  if (!track?.available || !track.totals || !track.days.length) {
    return (
      <>
        <PageHead eyebrow={t("Track record")} title={t("No live days scored yet")} />
        <Empty title={t("Waiting for the first results")} body={t("A forecast counts as live only if it was made before the 12:00 CET auction. Each one is scored when the real prices are published.")} />
      </>
    );
  }
  const T = track.totals, days = track.days;
  const skill = T.skill_vs_naive ?? 0, cov = T.coverage_10_90 ?? 0;
  const few = days.length < 30;
  const labels = days.map((d) => date(d.delivery_day, { day: "numeric", month: "short" }));
  return (
    <>
      <PageHead eyebrow={t("Track record · {0} live days", days.length)}
        title={skill >= 0 ? t("{0}% less error than 'same as yesterday', so far", n(skill * 100)) : t("{0}% more error than 'same as yesterday', so far", n(-skill * 100))}
        lede={t("Every forecast is logged before the auction closes and scored against the published price. Nothing is edited afterwards, and bad days stay in.")} />

      {few && <div className="mb-4"><Notice tone="warn">{t("{0} days is a small sample. One unusual week can move these numbers a lot; they become meaningful after a few months.", days.length)}</Notice></div>}

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label={t("Totals")}>
        <Stat label={t("Forecast error")} value={`€${n(T.mae_model, 1)}`} sub={t("average miss per MWh")} />
        <Stat label={t("'Same as yesterday' error")} value={`€${n(T.mae_naive, 1)}`} sub={t("the free baseline")} />
        <Stat label={t("Days the forecast won")} value={`${T.days_model_beat_naive} / ${days.length}`} />
        <Stat label={t("Price inside the range")} value={`${n(cov * 100)}%`} sub={t("target: 80%")} tone={cov < 0.7 ? "warn" : cov > 0.9 ? "warn" : "good"} />
      </section>

      <section className="card card-pad mt-4" aria-label={t("Error by day")}>
        <h2 className="mb-1 text-lg font-bold">{t("Average miss per day")}</h2>
        <p className="mb-4 text-sm text-text-2">{t("Lower is better. Where the grey bar is shorter, the simple rule beat the model that day.")}</p>
        <PairBars labels={labels} ariaLabel={t("Average forecast error per live day, model against the naive baseline.")}
          a={{ label: t("Forecast"), color: "var(--series)", values: days.map((d) => d.mae_model) }}
          b={{ label: t("Same as yesterday"), color: "var(--neutral)", values: days.map((d) => d.mae_naive) }} />
        <TableToggle>
          <table className="data">
            <thead><tr><th>{t("Day")}</th><th className="n">{t("Avg price")}</th><th className="n">{t("Forecast error")}</th><th className="n">{t("Naive error")}</th><th className="n">{t("Bias")}</th><th className="n">{t("In range")}</th></tr></thead>
            <tbody>
              {[...days].reverse().map((d) => (
                <tr key={d.delivery_day}><td>{date(d.delivery_day, { weekday: "short", day: "numeric", month: "short" })}</td><td className="n">{n(d.avg_price)}</td>
                  <td className="n font-semibold" style={{ color: d.mae_model <= d.mae_naive ? "var(--good)" : "var(--crit)" }}>{n(d.mae_model, 1)}</td>
                  <td className="n">{n(d.mae_naive, 1)}</td><td className="n">{d.bias > 0 ? "+" : ""}{n(d.bias, 1)}</td><td className="n">{n(d.coverage_10_90 * 100)}%</td></tr>
              ))}
            </tbody>
          </table>
        </TableToggle>
      </section>

      <p className="mt-4 text-sm text-muted">{t("Bias: positive means prices came in above the forecast. In range: share of hours where the price fell inside the 10–90% band.")}</p>
    </>
  );
}
