import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { authMode, COOKIE, safeEqual, sessionToken } from "@/lib/auth";

export async function proxy(request: NextRequest) {
  const mode = authMode();
  if (mode === "open") return NextResponse.next();
  const { pathname } = request.nextUrl;
  if (mode === "closed") {
    return new NextResponse("Sign-in is not configured for this deployment. Set WEB_PASSWORD.", { status: 503 });
  }
  if (pathname === "/login" || pathname.startsWith("/api/login")) return NextResponse.next();
  const expected = await sessionToken();
  const got = request.cookies.get(COOKIE)?.value ?? "";
  if (expected && safeEqual(got, expected)) return NextResponse.next();
  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = pathname && pathname !== "/" ? `?next=${encodeURIComponent(pathname)}` : "";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
