import type { Metadata } from "next";
import { BatteryView } from "@/components/BatteryView";
import { load } from "@/lib/data";
import type { Forecast } from "@/lib/types";

export const metadata: Metadata = { title: "Battery plan" };

export default async function Page() {
  const [forecast] = await Promise.all([load<Forecast>("forecast_latest")]);
  return <BatteryView forecast={forecast} />;
}
