import { TomorrowView } from "@/components/TomorrowView";
import { load } from "@/lib/data";
import type { Forecast, TrackRecord } from "@/lib/types";

export default async function Page() {
  const [forecast, track] = await Promise.all([load<Forecast>("forecast_latest"), load<TrackRecord>("track_record")]);
  return <TomorrowView forecast={forecast} track={track} />;
}
