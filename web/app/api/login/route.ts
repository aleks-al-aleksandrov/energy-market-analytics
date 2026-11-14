import { NextResponse } from "next/server";
import { authMode, COOKIE, MAX_AGE, passwordMatches, sessionToken } from "@/lib/auth";

function safeNext(value: FormDataEntryValue | null): string {
  const v = typeof value === "string" ? value : "";
  return v.startsWith("/") && !v.startsWith("//") && !v.includes("\\") ? v : "/";       // same-site paths only
}

export async function POST(request: Request) {
  const form = await request.formData();
  const next = safeNext(form.get("next"));
  const base = new URL(request.url);
  if (authMode() !== "password") return NextResponse.redirect(new URL("/", base), 303);
  const ok = await passwordMatches(String(form.get("password") ?? ""));
  if (!ok) {
    await new Promise((r) => setTimeout(r, 700));                                        // slow down guessing
    const back = new URL("/login", base);
    back.searchParams.set("error", "1");
    if (next !== "/") back.searchParams.set("next", next);
    return NextResponse.redirect(back, 303);
  }
  const res = NextResponse.redirect(new URL(next, base), 303);
  res.cookies.set(COOKIE, (await sessionToken()) ?? "", {
    httpOnly: true, sameSite: "lax", secure: base.protocol === "https:", path: "/", maxAge: MAX_AGE,
  });
  return res;
}

export async function DELETE(request: Request) {
  const res = NextResponse.redirect(new URL("/login", request.url), 303);
  res.cookies.delete(COOKIE);
  return res;
}
