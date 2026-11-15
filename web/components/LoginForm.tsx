"use client";

import { useState } from "react";
import { usePrefs } from "@/lib/prefs";

export function LoginForm({ next, error }: { next: string; error: boolean }) {
  const { t } = usePrefs();
  const [show, setShow] = useState(false);
  return (
    <div className="mx-auto mt-[8vh] w-full max-w-sm">
      <form method="post" action="/api/login" className="card card-pad fade-in grid gap-4">
        <div>
          <div className="eyebrow">BG Power</div>
          <h1 className="mt-1 text-2xl font-extrabold tracking-tight">{t("Sign in")}</h1>
          <p className="mt-2 text-sm text-text-2">{t("Enter the access password to open the dashboard.")}</p>
        </div>
        <input type="hidden" name="next" value={next} />
        <div className="grid gap-1.5">
          <label htmlFor="password" className="text-sm font-semibold text-text-2">{t("Password")}</label>
          <div className="relative">
            <input id="password" name="password" type={show ? "text" : "password"} required autoFocus autoComplete="current-password"
              aria-invalid={error || undefined} aria-describedby={error ? "login-error" : undefined}
              className="h-12 w-full rounded-xl border border-border bg-surface-2 pl-4 pr-24 text-base text-text outline-none focus:border-accent" />
            <button type="button" onClick={() => setShow((v) => !v)} aria-pressed={show}
              className="absolute right-1 top-1 h-10 rounded-lg px-3 text-sm font-semibold text-text-2 hover:text-text">{show ? t("Hide") : t("Show")}</button>
          </div>
        </div>
        {error && <p id="login-error" role="alert" className="text-sm font-semibold" style={{ color: "var(--crit)" }}>{t("Wrong password. Try again.")}</p>}
        <button type="submit" className="btn btn-primary">{t("Sign in")}</button>
      </form>
    </div>
  );
}
