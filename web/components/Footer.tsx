"use client";

import { usePrefs } from "@/lib/prefs";

export function Footer() {
  const { t } = usePrefs();
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-8 text-sm text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>{t("Forecasts and analysis only. This is not trading or investment advice.")}</p>
        <p>{t("Data: ENTSO-E Transparency Platform, Open-Meteo.")}</p>
      </div>
    </footer>
  );
}
