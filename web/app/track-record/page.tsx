import type { Metadata } from "next";
import { TrackView } from "@/components/TrackView";
import { load } from "@/lib/data";
import type { TrackRecord } from "@/lib/types";

export const metadata: Metadata = { title: "Track record" };

export default async function Page() {
  const [track] = await Promise.all([load<TrackRecord>("track_record")]);
  return <TrackView track={track} />;
}
