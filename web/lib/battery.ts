// One-day battery schedule on a price curve, solved in the browser by dynamic programming.
// Same model as src/battery.py (the Python version is a linear program): power limit, energy capacity, starts and ends empty,
// round-trip efficiency split as sqrt(eta) on each side, a cap on charged energy per day (cycles), optional wear cost.
// State = (state of charge, energy charged so far) on a grid; the grid makes it approximate (about 1 % below the LP).

export type BatteryParams = { powerMw: number; durationH: number; efficiency: number; cyclesPerDay: number; degradationEurMwh: number };
export type Plan = { net: number[]; soc: number[]; revenue: number; charged: number; discharged: number };

const STEPS = 96;      // state-of-charge grid points per full battery (about 1 % below the exact LP, on the cautious side)

export function planDay(prices: number[], p: BatteryParams): Plan {
  const n = prices.length;
  const E = p.powerMw * p.durationH;
  const se = Math.sqrt(p.efficiency);
  const unit = E / STEPS;                                  // stored energy per grid step (MWh)
  const maxStepPerHour = Math.max(1, Math.floor((p.powerMw * se) / unit + 1e-9));       // charging: stored = grid * se
  const maxDisPerHour = Math.max(1, Math.floor((p.powerMw / se) / unit + 1e-9));        // discharging: stored drop = sold / se
  const budget = Math.round(STEPS * p.cyclesPerDay);       // total stored-energy steps that may be charged in the day
  const B = budget + 1, S = STEPS + 1;
  const NEG = -1e18;
  // value[t][s][c] = best revenue from hour t on, with s steps stored and c steps of charge budget used
  let next = new Float64Array(S * B).fill(NEG);
  for (let c = 0; c < B; c++) next[0 * B + c] = 0;         // must end empty
  const choice: Int16Array[] = [];
  for (let t = n - 1; t >= 0; t--) {
    const cur = new Float64Array(S * B).fill(NEG);
    const ch = new Int16Array(S * B);
    const price = prices[t];
    for (let s = 0; s < S; s++) {
      for (let c = 0; c < B; c++) {
        let best = NEG, bestA = 0;
        const lo = -Math.min(maxDisPerHour, s), hi = Math.min(maxStepPerHour, STEPS - s, budget - c);
        for (let a = lo; a <= hi; a++) {                   // a > 0 charge (steps stored), a < 0 discharge
          const fut = next[(s + a) * B + (a > 0 ? c + a : c)];
          if (fut <= NEG / 2) continue;
          const stored = Math.abs(a) * unit;
          const gridMwh = a > 0 ? stored / se : stored * se;       // bought from / sold to the grid
          const cash = a > 0 ? -gridMwh * price : gridMwh * price;
          const v = cash - (p.degradationEurMwh * gridMwh) / 2 + fut;
          if (v > best) { best = v; bestA = a; }
        }
        cur[s * B + c] = best; ch[s * B + c] = bestA;
      }
    }
    choice[t] = ch; next = cur;
  }
  const net: number[] = [], soc: number[] = [];
  let s = 0, c = 0, charged = 0, discharged = 0;
  for (let t = 0; t < n; t++) {
    const a = choice[t][s * B + c];
    const stored = Math.abs(a) * unit;
    const gridMwh = a > 0 ? stored / se : stored * se;
    net.push(a > 0 ? -gridMwh : a < 0 ? gridMwh : 0);
    if (a > 0) { charged += gridMwh; c += a; } else if (a < 0) discharged += gridMwh;
    s += a; soc.push(s * unit);
  }
  const revenue = Math.max(0, next[0]);
  return { net, soc, revenue, charged, discharged };
}

/** Revenue of a fixed schedule if prices turn out differently (wear cost included). */
export function settle(net: number[], prices: number[], degradationEurMwh = 0): number {
  let r = 0, through = 0;
  for (let i = 0; i < net.length; i++) { r += net[i] * prices[i]; through += Math.abs(net[i]); }
  return r - (degradationEurMwh * through) / 2;
}
