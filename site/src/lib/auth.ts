import { createHmac, timingSafeEqual } from "crypto";

const PAYLOAD = "ok";

function sign(payload: string, secret: string): string {
  return createHmac("sha256", secret).update(payload).digest("hex");
}

export function issueCookieValue(secret: string): string {
  return `${PAYLOAD}.${sign(PAYLOAD, secret)}`;
}

export function isValidCookie(value: string | undefined, secret: string): boolean {
  if (!value) return false;
  const [payload, sig] = value.split(".");
  if (payload !== PAYLOAD || !sig) return false;
  const expected = sign(PAYLOAD, secret);
  const a = Buffer.from(sig);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

export function checkPassword(input: string): boolean {
  const expected = process.env.LAB_PASSWORD ?? "";
  if (!expected) return false;
  const a = Buffer.from(input);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}
