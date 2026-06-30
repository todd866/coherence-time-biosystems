import { NextResponse, type NextRequest } from "next/server";
import { checkPassword, issueCookieValue } from "@/lib/auth";

export async function POST(req: NextRequest) {
  const secret = process.env.AUTH_SECRET ?? "";
  if (!secret) {
    return NextResponse.json(
      { ok: false, error: "server misconfigured" },
      { status: 500 },
    );
  }

  const { password } = await req.json();
  if (!checkPassword(password ?? "")) {
    return NextResponse.json({ ok: false }, { status: 401 });
  }

  const res = NextResponse.json({ ok: true });
  res.cookies.set("bayes_session", issueCookieValue(secret), {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return res;
}

export async function DELETE() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set("bayes_session", "", { path: "/", maxAge: 0 });
  return res;
}
