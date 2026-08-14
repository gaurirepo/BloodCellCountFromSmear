const ENGINE = process.env.ENGINE_URL ?? "http://127.0.0.1:8000";

export const maxDuration = 30;

export async function GET() {
  try {
    const response = await fetch(`${ENGINE}/health`, { cache: "no-store" });
    const body = await response.text();
    return new Response(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    return Response.json({ status: "offline" }, { status: 503 });
  }
}
