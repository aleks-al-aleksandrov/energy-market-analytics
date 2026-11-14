// Access gate for the hosted app. One shared password, kept in the WEB_PASSWORD environment variable (never in the code).
//   WEB_PASSWORD set           -> every page needs a signed session cookie; /login sets it.
//   WEB_AUTH_REQUIRED=1, no pw -> fail closed: nothing is served (a forgotten secret must not leave the app open).
//   neither                    -> open (local development).
// The cookie holds HMAC-SHA256(WEB_PASSWORD, fixed message): it cannot be forged without the password, and changing the
// password signs everyone out. This keeps casual visitors out; it is not a substitute for real user accounts.

export const COOKIE = "ema_auth";
export const MAX_AGE = 60 * 60 * 24 * 30;
const MESSAGE = "bg-power-session-v1";

export function authMode(): "open" | "password" | "closed" {
  if (process.env.WEB_PASSWORD) return "password";
  return process.env.WEB_AUTH_REQUIRED === "1" ? "closed" : "open";
}

async function hmacHex(key: string, message: string): Promise<string> {
  const enc = new TextEncoder();
  const k = await crypto.subtle.importKey("raw", enc.encode(key), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", k, enc.encode(message));
  return Array.from(new Uint8Array(sig)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function sessionToken(): Promise<string | null> {
  const pw = process.env.WEB_PASSWORD;
  return pw ? hmacHex(pw, MESSAGE) : null;
}

/** Constant-time string comparison (both are hex digests of equal length when valid). */
export function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** Compare a submitted password with the configured one without leaking length or position through timing. */
export async function passwordMatches(submitted: string): Promise<boolean> {
  const pw = process.env.WEB_PASSWORD;
  if (!pw) return false;
  const [a, b] = await Promise.all([hmacHex(pw, submitted), hmacHex(pw, pw)]);
  return safeEqual(a, b);
}
