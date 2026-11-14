"use client";

import { usePrefs } from "@/lib/prefs";
import type { Forecast, Metrics } from "@/lib/types";
import { Notice, PageHead } from "./ui";

const NAME: Record<string, string> = {
  lgbm_monthly: "LightGBM, monthly refit (previous model)", lgbm_weekly: "LightGBM, weekly refit", lgbm_asinh: "LightGBM on asinh prices",
  lgbm_asinh_weekly: "LightGBM on asinh prices, weekly refit", lear_weekly: "LEAR (per-hour lasso)", ada3: "Adaptive ensemble of three",
  ada4: "Adaptive ensemble of four (in production)", ridge_weekly: "Ridge, weekly refit", level_weekly: "Level-adjusted LightGBM, weekly refit",
};
const MEMBER: Record<string, string> = { lightgbm: "LightGBM", ridge: "Ridge", level: "Level-adjusted LightGBM", lear: "LEAR" };

export function ModelView({ metrics, forecast }: { metrics: Metrics | null; forecast: Forecast | null }) {
  const { t, n } = usePrefs();
  const p1 = metrics?.phase1 ?? [];
  const order = ["ada4", "lear_weekly", "lgbm_asinh_weekly", "lgbm_weekly", "lgbm_monthly"];
  const rows = order.map((m) => ({ m, cv: p1.find((r) => r.model === m && r.split === "cv"), ho: p1.find((r) => r.model === m && r.split === "holdout") })).filter((r) => r.cv);
  const bands = (metrics?.band_calibration ?? []).filter((b) => b.chosen || b.variant.startsWith("raw") || b.variant === "conformal 180d");
  const bandNames = [...new Set(bands.map((b) => b.variant))];
  const weights = forecast?.ensemble?.weights;
  const best = rows[0]?.cv, base = rows.find((r) => r.m === "lgbm_monthly")?.cv;

  return (
    <>
      <PageHead eyebrow={t("Model")} title={t("How the forecast is made, and where it fails")}
        lede={t("Four models forecast tomorrow's 24 prices using only what is known before the 12:00 CET auction. They are blended by how well each did over the last 14 days, so when prices jump to a new level the models that can follow get the weight.")} />

      <section className="grid gap-4 lg:grid-cols-3">
        {[
          [t("1. Only what is known in time"), t("Yesterday's prices here and in neighbouring countries, the grid operator's forecasts of demand, wind and solar, the weather forecast, and the calendar. Nothing from the delivery day itself.")],
          [t("2. Tested the way it is used"), t("Each model is trained on the past and scored on the following days, repeated across 25 months. Changes are accepted only by rules fixed before looking at the result.")],
          [t("3. Scored in public"), t("Every live forecast is logged before the auction and compared with the published price. The track record page shows all of it, including the bad days.")],
        ].map(([h, b]) => (<div key={h} className="card card-pad"><h2 className="text-base font-bold">{h}</h2><p className="mt-2 text-sm leading-relaxed text-text-2">{b}</p></div>))}
      </section>

      {weights && (
        <section className="card card-pad mt-4" aria-label={t("Today's blend")}>
          <h2 className="text-lg font-bold">{t("Today's blend")}</h2>
          <p className="mt-1 text-sm text-text-2">{t("Share of each model in tomorrow's forecast, set by its error over the last 14 published days.")}</p>
          <div className="mt-4 grid gap-3">
            {Object.entries(weights).sort((a, b) => b[1] - a[1]).map(([k, w]) => (
              <div key={k} className="grid grid-cols-[minmax(120px,220px)_1fr_48px] items-center gap-3 text-sm">
                <span className="font-semibold">{MEMBER[k] ?? k}</span>
                <span className="h-2.5 overflow-hidden rounded-full bg-surface-2" aria-hidden><span className="block h-full rounded-full" style={{ width: `${w * 100}%`, background: "var(--series)" }} /></span>
                <span className="num text-right font-bold">{n(w * 100)}%</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {rows.length > 0 && (
        <section className="card card-pad mt-4" aria-label={t("Back-test results")}>
          <h2 className="text-lg font-bold">{best && base ? t("The blend cut the average error from €{0} to €{1}", n(base.MAE, 1), n(best.MAE, 1)) : t("Back-test results")}</h2>
          <p className="mt-1 text-sm text-text-2">{t("Average miss in EUR/MWh over the same hours for every model. Skill is the reduction in error against 'same hour yesterday'.")}</p>
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="data">
              <thead><tr><th>{t("Model")}</th><th className="n">{t("Error, 25 test months")}</th><th className="n">{t("Skill")}</th><th className="n">{t("Error, Aug–Oct 2026")}</th><th className="n">{t("Skill")}</th></tr></thead>
              <tbody>
                {rows.map(({ m, cv, ho }) => (
                  <tr key={m}><td className={m === "ada4" ? "font-bold" : ""}>{t(NAME[m] ?? m)}</td>
                    <td className="n font-semibold">{n(cv!.MAE, 2)}</td><td className="n">{n(cv!.skill_vs_naive * 100, 1)}%</td>
                    <td className="n">{ho ? n(ho.MAE, 2) : "–"}</td><td className="n">{ho ? `${n(ho.skill_vs_naive * 100, 1)}%` : "–"}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4"><Notice tone="warn">{t("Aug–Oct 2026 was a price surge beyond anything in the training data. Those three months have been looked at many times while improving the model, so they no longer count as an untouched test. The clean evidence from here on is the live track record.")}</Notice></div>
        </section>
      )}

      {bandNames.length > 0 && (
        <section className="card card-pad mt-4" aria-label={t("Uncertainty range")}>
          <h2 className="text-lg font-bold">{t("How often the price lands inside the range")}</h2>
          <p className="mt-1 text-sm text-text-2">{t("The range should hold the price 80% of the time. The adaptive version widens itself after misses and narrows when it is too cautious.")}</p>
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="data">
              <thead><tr><th>{t("Range")}</th><th className="n">{t("Inside, 25 test months")}</th><th className="n">{t("Inside, Aug–Oct 2026")}</th><th className="n">{t("Average width")}</th></tr></thead>
              <tbody>
                {bandNames.map((v) => {
                  const cv = bands.find((b) => b.variant === v && b.split === "cv"), ho = bands.find((b) => b.variant === v && b.split === "holdout");
                  const label = v.startsWith("raw") ? t("Raw model output") : v.startsWith("adaptive") ? t("Adaptive (in production)") : t("Fixed 180-day adjustment (previous)");
                  return (<tr key={v}><td className={cv?.chosen ? "font-bold" : ""}>{label}</td><td className="n">{cv ? `${n(cv.coverage * 100, 1)}%` : "–"}</td><td className="n">{ho ? `${n(ho.coverage * 100, 1)}%` : "–"}</td><td className="n">€{n(cv?.width)}</td></tr>);
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="card card-pad mt-4" aria-label={t("Tested and rejected")}>
        <h2 className="text-lg font-bold">{t("Tested and rejected")}</h2>
        <p className="mt-1 text-sm text-text-2">{t("Ideas that sounded right and did not improve the forecast under the same rule. They stay out.")}</p>
        <ul className="mt-3 grid gap-2 text-sm leading-relaxed text-text-2">
          <li>• {t("Gas, oil and carbon prices: yesterday's power price already carries the fuel level.")}</li>
          <li>• {t("Interconnector capacities: informative alone, no gain on top of price lags.")}</li>
          <li>• {t("Plant outages: the public archive keeps only the latest revision, so history cannot be tested without leaking the future.")}</li>
          {metrics?.regional_choice && !metrics.regional_choice.adopt && (metrics.regional_choice.held_back
            ? <li>• {t("German prices and neighbours' demand and renewable forecasts: better in normal months (error {0} against {1}) but much worse in the 2026 price surge ({2} against {3}), so they are held back.", n(metrics.regional_choice.cv_mae_with, 1), n(metrics.regional_choice.cv_mae_without, 1), n(metrics.regional_choice.holdout_mae_with, 1), n(metrics.regional_choice.holdout_mae_without, 1))}</li>
            : <li>• {t("German prices and neighbours' demand and renewable forecasts: error {0} with them, {1} without.", n(metrics.regional_choice.cv_mae_with, 2), n(metrics.regional_choice.cv_mae_without, 2))}</li>)}
          {metrics?.battery_uncertainty_choice?.schedule_on === "forecast" && <li>• {t("Correcting the forecast's hourly bias before scheduling a battery: revenue changed by less than the noise.")}</li>}
        </ul>
        {metrics?.regional_choice?.adopt && <p className="mt-3 text-sm text-text-2">{t("Adopted: German prices and neighbours' forecasts lowered the error from {0} to {1}.", n(metrics.regional_choice.cv_mae_without, 2), n(metrics.regional_choice.cv_mae_with, 2))}</p>}
      </section>

      <section className="card card-pad mt-4" aria-label={t("Limits")}>
        <h2 className="text-lg font-bold">{t("Limits")}</h2>
        <ul className="mt-3 grid gap-2 text-sm leading-relaxed text-text-2">
          <li>• {t("Forecasts are hourly averages; the market now clears in 15-minute steps.")}</li>
          <li>• {t("Sudden regime changes are still the weak point: the blend adapts within days, not hours.")}</li>
          <li>• {t("The battery figures cover day-ahead arbitrage only and assume the battery does not move the price.")}</li>
        </ul>
      </section>
    </>
  );
}
