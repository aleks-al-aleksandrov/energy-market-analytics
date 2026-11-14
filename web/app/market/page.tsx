import type { Metadata } from "next";
import { MarketView } from "@/components/MarketView";
import { load } from "@/lib/data";
import type { Market } from "@/lib/types";

export const metadata: Metadata = { title: "Market" };

export default async function Page() {
  const [market] = await Promise.all([load<Market>("market_recent")]);
  return <MarketView market={market} />;
}
