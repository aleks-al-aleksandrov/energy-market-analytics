# BG Power web app

The public face of the project: a Next.js app that reads the JSON payloads the Python pipeline publishes every day
(`reports/api/v1/*.json`). It has no database and no model code; if the daily run stops, the app says so.

| Page | What it shows |
|---|---|
| `/` Tomorrow | Headline computed from the forecast, the hourly chart with its 10–90 % range and a 15-minute view, five plain questions that light up the chart, an hour table |
| `/battery` | Interactive battery plan: power, duration, efficiency, cycles and wear cost; solved in the browser |
| `/track-record` | Every live forecast scored against the published price and against "same as yesterday" |
| `/market` | Recent day-ahead prices for Bulgaria, Romania, Greece and Hungary |
| `/model` | How the forecast is made, back-test results, today's model blend, what was tested and rejected, limits |

English and Bulgarian, dark and light, phone to desktop. Every chart has a table view.

## Run locally

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

In development the app reads `../reports/api/v1` directly. Regenerate those files with `python -m src.export_api`.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `DATA_BASE_URL` | Where the JSON lives. Default: this repository's raw files on GitHub (refreshed by the daily workflow). |
| `DATA_SOURCE=remote` | Always fetch from `DATA_BASE_URL`, never the local folder. |
| `DATA_TOKEN` | Read-only GitHub token, only needed if the repository is private. |
| `WEB_PASSWORD` | Turns on the sign-in page. One shared password; changing it signs everyone out. |
| `WEB_AUTH_REQUIRED=1` | Fail closed: without `WEB_PASSWORD` nothing is served. Set this on any hosted deployment. |

## Deploy on Vercel

1. Import the GitHub repository in Vercel and set **Root Directory** to `web`.
2. Add `WEB_PASSWORD` and `WEB_AUTH_REQUIRED=1` under *Settings → Environment Variables*.
3. Deploy. Pages re-read the JSON at most every 10 minutes, so a new daily forecast appears without a redeploy.

The password gate keeps casual visitors out. It does not hide data that is in a public repository: if the numbers must stay
private, make the repository private and set `DATA_TOKEN`.

## Notes for contributors

* Interface text goes through `t("English source")`; Bulgarian lives in `lib/i18n-bg.ts` and falls back to English.
* Colours are CSS variables in `app/globals.css`; the chart palettes are the ones validated for colour-blind separation in
  the Streamlit app.
* `lib/battery.ts` mirrors `src/battery.py` (dynamic programming instead of a linear program; within about 1 %).
* This Next.js version differs from older ones: read `node_modules/next/dist/docs/` before changing conventions
  (for example, request interception lives in `proxy.ts`, not `middleware.ts`).
