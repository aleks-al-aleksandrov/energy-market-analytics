"use client";

import Link from "next/link";
import { usePrefs } from "@/lib/prefs";

/** Skeleton shown while a page's data loads (same outline as the real page, so nothing jumps when it arrives). */
export function LoadingState() {
  const { t } = usePrefs();
  return (
    <div role="status" aria-live="polite">
      <span className="sr-only">{t("Loading the latest forecast")}</span>
      <div className="skeleton h-4 w-40" />
      <div className="skeleton mt-3 h-10 w-full max-w-2xl" />
      <div className="skeleton mt-3 h-5 w-full max-w-xl" />
      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => <div key={i} className="skeleton h-28" />)}
      </div>
      <div className="skeleton mt-4 h-[380px]" />
    </div>
  );
}

export function ErrorState({ retry }: { retry: () => void }) {
  const { t } = usePrefs();
  return (
    <div className="card card-pad mx-auto mt-10 max-w-lg text-center" role="alert">
      <h1 className="text-xl font-bold">{t("Something went wrong")}</h1>
      <p className="mt-2 text-text-2">{t("The page could not load its data. This is usually temporary.")}</p>
      <button className="btn btn-primary mt-5" onClick={retry}>{t("Try again")}</button>
    </div>
  );
}

export function NotFoundState() {
  const { t } = usePrefs();
  return (
    <div className="card card-pad mx-auto mt-10 max-w-lg text-center">
      <h1 className="text-xl font-bold">{t("Page not found")}</h1>
      <p className="mt-2 text-text-2">{t("That address does not exist. The forecast is on the home page.")}</p>
      <Link href="/" className="btn btn-primary mt-5">{t("Back to tomorrow's forecast")}</Link>
    </div>
  );
}
