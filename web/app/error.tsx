"use client";

import { ErrorState } from "@/components/States";

export default function Error({ reset }: { error: Error; reset: () => void }) {
  return <ErrorState retry={reset} />;
}
