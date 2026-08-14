const ENGINE = process.env.ENGINE_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  const form = await request.formData();

  try {
    const response = await fetch(`${ENGINE}/analyze`, {
      method: "POST",
      body: form,
    });
    const body = await response.text();
    return new Response(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    return Response.json(
      { detail: "Cannot reach the inference engine." },
      { status: 503 },
    );
  }
}
