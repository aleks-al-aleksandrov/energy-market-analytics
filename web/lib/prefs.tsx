"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { BG } from "./i18n-bg";

export type Lang = "en" | "bg";
export type Theme = "dark" | "light";

type Prefs = {
  lang: Lang; theme: Theme; setLang: (l: Lang) => void; setTheme: (t: Theme) => void;
  /** Translate an English source string; {0}, {1}... are replaced by the arguments. Missing Bulgarian falls back to English. */
  t: (en: string, ...args: (string | number)[]) => string;
  /** Locale-aware number. */
  n: (value: number | null | undefined, digits?: number) => string;
  date: (iso: string, opts?: Intl.DateTimeFormatOptions) => string;
};

const Ctx = createContext<Prefs | null>(null);

function store(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* private mode: preference lasts for this visit only */ }
  document.cookie = `${key}=${value}; path=/; max-age=31536000; samesite=lax`;
}

export function PrefsProvider({ initialLang, initialTheme, children }: { initialLang: Lang; initialTheme: Theme; children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const [theme, setThemeState] = useState<Theme>(initialTheme);

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);

  const setLang = useCallback((l: Lang) => { setLangState(l); store("ema_lang", l); }, []);
  const setTheme = useCallback((t: Theme) => { setThemeState(t); store("ema_theme", t); }, []);

  const value = useMemo<Prefs>(() => {
    const locale = lang === "bg" ? "bg-BG" : "en-GB";
    return {
      lang, theme, setLang, setTheme,
      t: (en, ...args) => {
        const s = lang === "bg" ? (BG[en] ?? en) : en;
        return args.length ? s.replace(/\{(\d+)\}/g, (_, i) => String(args[Number(i)] ?? "")) : s;
      },
      n: (v, digits = 0) => (v === null || v === undefined || Number.isNaN(v))
        ? "–" : new Intl.NumberFormat(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v),
      date: (iso, opts = { weekday: "long", day: "numeric", month: "long" }) =>
        new Intl.DateTimeFormat(locale, { timeZone: "Europe/Sofia", ...opts }).format(new Date(iso.length === 10 ? `${iso}T12:00:00Z` : iso)),
    };
  }, [lang, theme, setLang, setTheme]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function usePrefs(): Prefs {
  const c = useContext(Ctx);
  if (!c) throw new Error("usePrefs must be used inside PrefsProvider");
  return c;
}
