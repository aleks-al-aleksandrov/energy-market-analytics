"use client";

import { useMemo, useState } from "react";
import { planDay, settle } from "@/lib/battery";
import { usePrefs } from "@/lib/prefs";
import type { Forecast } from "@/lib/types";
import { ScheduleChart } from "./charts";
import { Empty, Notice, PageHead, Stat, TableToggle } from "./ui";

function Slider({ label, value, min, max, step, unit, onChange, hint }: { label: string; value: number; min: number; max: number; step: number; unit: string; onChange: (v: number) => void; hint?: string }) {
  const id = label.replace(/\W+/g, "-").toLowerCase();
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-sm font-semibold text-text-2">{label}</label>
        <output htmlFor={id} className="num text-base font-bold">{value}{unit}</output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <p className="text-xs text-muted">{hint}</p>}
    </div>
  );
}

export function BatteryView({ forecast }: { forecast: Forecast | null }) {
  const { t, n, date } = usePrefs();
  const [power, setPower] = useState(1);
  const [duration, setDuration] = useState(2);
  const [eff, setEff] = useState(90);
  const [cycles, setCycles] = useState(1);
  const [wear, setWear] = useState(0);

  const r = useMemo(() => {
    if (!forecast?.available) return null;
    const prices = forecast.hours.map((h) => h.forecast);
    const plan = planDay(prices, { powerMw: power, durationH: duration, efficiency: eff / 100, cyclesPerDay: cycles, degradationEurMwh: wear });
    return {
      prices, plan,
      low: settle(plan.net, forecast.hours.map((h) => h.p10), wear),
      high: settle(plan.net, forecast.hours.map((h) => h.p90), wear),
    };
  }, [forecast, power, duration, eff, cycles, wear]);

  if (!forecast?.available || !r) return <Empty title={t("No forecast yet")} body={t("The battery plan needs tomorrow's forecast.")} />;
  const hours = forecast.hours.map((h) => h.hour_local);
  const idle = r.plan.revenue < 0.5;
  const perMw = r.plan.revenue / power;

  return (
    <>
      <PageHead eyebrow={t("Battery · {0}", date(forecast.delivery_day))}
        title={idle ? t("Tomorrow's spread does not cover the losses") : t("About €{0} from one day of arbitrage", n(r.plan.revenue))}
        lede={t("Set your battery. The plan charges in the cheapest forecast hours and sells in the dearest, within power, capacity and efficiency limits. It is recalculated instantly in your browser.")} />

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <section className="card card-pad grid content-start gap-5" aria-label={t("Battery settings")}>
          <Slider label={t("Power")} value={power} min={0.5} max={20} step={0.5} unit=" MW" onChange={setPower} />
          <Slider label={t("Duration")} value={duration} min={1} max={6} step={0.5} unit=" h" onChange={setDuration} hint={t("Capacity: {0} MWh", n(power * duration, 1))} />
          <Slider label={t("Round-trip efficiency")} value={eff} min={70} max={98} step={1} unit="%" onChange={setEff} />
          <Slider label={t("Cycles per day")} value={cycles} min={1} max={2} step={0.5} unit="" onChange={setCycles} />
          <Slider label={t("Wear cost")} value={wear} min={0} max={30} step={1} unit=" €/MWh" onChange={setWear} hint={t("Cost of battery ageing per MWh moved.")} />
        </section>

        <div className="grid min-w-0 content-start gap-4">
          <section className="grid gap-3 sm:grid-cols-3" aria-label={t("Expected result")}>
            <Stat label={t("If prices match the forecast")} value={`€${n(r.plan.revenue)}`} sub={t("€{0} per MW", n(perMw))} />
            <Stat label={t("If prices land at the low edge")} value={`€${n(r.low)}`} sub={t("same schedule, 10% band edge")} tone={r.low < 0 ? "crit" : undefined} />
            <Stat label={t("If prices land at the high edge")} value={`€${n(r.high)}`} sub={t("same schedule, 90% band edge")} />
          </section>
          <section className="card card-pad" aria-label={t("Schedule")}>
            <ScheduleChart hours={hours} prices={r.prices} net={r.plan.net} soc={r.plan.soc} energy={power * duration} />
            <TableToggle>
              <table className="data">
                <thead><tr><th>{t("Hour")}</th><th className="n">{t("Price")}</th><th>{t("Action")}</th><th className="n">MW</th><th className="n">{t("Stored")} (MWh)</th></tr></thead>
                <tbody>
                  {hours.map((h, i) => (
                    <tr key={i}><td className="num">{String(h).padStart(2, "0")}:00</td><td className="n">{n(r.prices[i], 1)}</td>
                      <td>{r.plan.net[i] < -1e-6 ? t("Charge") : r.plan.net[i] > 1e-6 ? t("Sell") : t("Idle")}</td>
                      <td className="n">{n(Math.abs(r.plan.net[i]), 2)}</td><td className="n">{n(r.plan.soc[i], 2)}</td></tr>
                  ))}
                </tbody>
              </table>
            </TableToggle>
          </section>
          <Notice tone="warn">
            {t("Read this as an upper estimate. In back-tests a forecast-guided battery earned about 88% of what perfect foresight would, and realised revenue is usually below the forecast figure, because a schedule sells where the forecast is highest. Day-ahead arbitrage only: no grid fees, taxes, intraday or balancing income. The in-browser solver is within about 1% of the exact optimum.")}
          </Notice>
        </div>
      </div>
    </>
  );
}
