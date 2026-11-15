// Shapes of the payloads written by src/serving.py (reports/api/v1/*.json).

export type ForecastHour = {
  timestamp_utc: string; time_local: string; hour_local: number;
  forecast: number; p10: number; p90: number; spike_probability: number | null; cross_check: number | null;
  minute?: number;
};

export type Forecast = {
  available: boolean; reason?: string;
  delivery_day: string; unit: string; timezone: string;
  generated_at: string | null; age_hours: number | null; stale: boolean; model_trained_until: string | null;
  band: { level: string; calibrated_as_of: string | null };
  summary: {
    mean: number; min: number; max: number; min_hour_local: number; max_hour_local: number; spread: number; mean_band_width: number;
    cheapest_3h: { start_local: string; mean: number }; dearest_3h: { start_local: string; mean: number };
    max_spike_probability: number | null; models_disagree_by: number | null; models_disagree: boolean;
  };
  hours: ForecastHour[];
  quarters?: { timestamp_utc: string; time_local: string; hour_local: number; minute: number; forecast: number; p10: number; p90: number }[] | null;
  ensemble?: { delivery_day: string; weights: Record<string, number>; window_days: number } | null;
  disclaimer: string;
};

export type TrackDay = {
  delivery_day: string; hours: number; mae_model: number; mae_naive: number; bias: number; coverage_10_90: number; avg_price: number;
};
export type TrackRecord = {
  available: boolean; days: TrackDay[]; n_days?: number; first_day?: string; last_day?: string;
  totals?: { mae_model: number; mae_naive: number; skill_vs_naive: number | null; coverage_10_90: number | null; days_model_beat_naive: number };
  note?: string;
};

export type MetricRow = { model: string; MAE: number; RMSE: number; skill_vs_best_naive: number | null; DM_p: number | null; significant_5pct: boolean | null; n_folds?: number };
export type BandRow = { variant: string; split: string; coverage: number; width: number; chosen: boolean; worst_30d_cv?: number | null; below?: number; above?: number };
export type Metrics = {
  available: boolean; cv?: MetricRow[]; holdout?: MetricRow[]; band_calibration?: BandRow[];
  phase1_choice?: { point_forecast: string; baseline: string; rule: string; candidates_passing: string[]; band: string; band_rule: string; holdout_note: string };
  phase1?: { model: string; split: string; MAE: number; skill_vs_naive: number; skill_lo: number | null; skill_hi: number | null; delta_vs_baseline: number | null; DM_p: number | null }[];
  battery_uncertainty_choice?: { schedule_on: string; candidates_passing: string[] };
  quarter_hour?: { candidate: string; split: string; shape_mae: number; price_mae: number; delta_vs_flat: number | null; DM_p: number | null }[];
  quarter_choice?: { shape: string; mean_abs_dev: number; band_coverage?: Record<string, { hourly_band_only: number; with_addon: number; n: number }> };
  feature_choice?: Record<string, unknown>; regional_choice?: { adopt: boolean; held_back?: boolean; cv_mae_without: number; cv_mae_with: number; holdout_mae_without?: number; holdout_mae_with?: number; n_features: number };
  caveat?: string;
};

export type Market = {
  available: boolean; unit?: string; zones?: string[]; data_through?: string;
  hourly?: ({ timestamp_utc: string } & Record<string, number | string | null>)[];
  daily?: ({ day: string } & Record<string, number | string | null>)[];
};

export type Status = {
  ok: boolean; problems: string[]; checked_at: string;
  forecast: { delivery_day: string | null; generated_at: string | null; age_hours: number | null; model_trained_until: string | null };
  market_data_through: string | null; band: { method: string; as_of: string | null; live_days_applied: number }; live_track_record_days: number;
};
