import "server-only";
import { promises as fs } from "fs";
import path from "path";

// Where the JSON payloads come from, in order:
//  1. DATA_LOCAL_DIR (or ../reports/api/v1 when it exists): local development next to the Python project.
//  2. DATA_BASE_URL: any static host. Default: the repository's raw files, refreshed by the daily workflow.
// For a private repository set DATA_TOKEN (a read-only GitHub token); it is sent only to DATA_BASE_URL.
const BASE = process.env.DATA_BASE_URL
  ?? "https://raw.githubusercontent.com/aleks-al-aleksandrov/energy-market-analytics/main/reports/api/v1";
const LOCAL = process.env.DATA_LOCAL_DIR ?? path.join(process.cwd(), "..", "reports", "api", "v1");
const REVALIDATE_SECONDS = 600;

async function readLocal<T>(name: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(path.join(LOCAL, `${name}.json`), "utf-8")) as T;
  } catch {
    return null;
  }
}

export async function load<T>(name: string): Promise<T | null> {
  if (process.env.DATA_SOURCE !== "remote") {
    const local = await readLocal<T>(name);
    if (local) return local;
  }
  try {
    const headers: Record<string, string> = {};
    if (process.env.DATA_TOKEN) headers.Authorization = `token ${process.env.DATA_TOKEN}`;
    const res = await fetch(`${BASE}/${name}.json`, { headers, next: { revalidate: REVALIDATE_SECONDS } });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}
