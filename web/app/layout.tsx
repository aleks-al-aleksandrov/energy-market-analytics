import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { cookies } from "next/headers";
import "./globals.css";
import { PrefsProvider, type Lang, type Theme } from "@/lib/prefs";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { load } from "@/lib/data";
import type { Status } from "@/lib/types";

const inter = Inter({ subsets: ["latin", "cyrillic"], variable: "--font-inter", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin", "cyrillic"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "BG Power · day-ahead price forecast", template: "%s · BG Power" },
  description: "Tomorrow's Bulgarian day-ahead electricity prices: hourly forecast, calibrated uncertainty, live track record and battery plan.",
};
export const viewport: Viewport = { themeColor: [{ color: "#0B0E14" }], width: "device-width", initialScale: 1 };

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const jar = await cookies();
  const lang: Lang = jar.get("ema_lang")?.value === "bg" ? "bg" : "en";
  const theme: Theme = jar.get("ema_theme")?.value === "light" ? "light" : "dark";
  const status = await load<Status>("status");
  return (
    <html lang={lang} data-theme={theme} className={`${inter.variable} ${mono.variable}`}>
      <body>
        <PrefsProvider initialLang={lang} initialTheme={theme}>
          <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 btn">Skip to content</a>
          <Header status={status} />
          <main id="main" className="mx-auto w-full max-w-6xl px-4 pb-20 pt-6 sm:px-6 sm:pt-10">{children}</main>
          <Footer />
        </PrefsProvider>
      </body>
    </html>
  );
}
