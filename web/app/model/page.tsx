import type { Metadata } from "next";
import { ModelView } from "@/components/ModelView";
import { load } from "@/lib/data";
import type { Forecast, Metrics } from "@/lib/types";

export const metadata: Metadata = { title: "Model" };

export default async function Page() {
  const [metrics, forecast] = await Promise.all([load<Metrics>("metrics"), load<Forecast>("forecast_latest")]);
  return <ModelView metrics={metrics} forecast={forecast} />;
}
