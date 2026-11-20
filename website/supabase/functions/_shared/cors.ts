const WEBSITE_ORIGINS = ['https://www.aidotnet.dev', 'https://test.aidotnet.dev'];

export function websiteOrigin(req: Request): string | undefined {
  const configured = [Deno.env.get('ALLOWED_ORIGIN'), Deno.env.get('ALLOWED_ORIGINS')]
    .filter((value): value is string => !!value).flatMap((value) => value.split(',').map((origin) => origin.trim()));
  const origin = req.headers.get('Origin');
  return origin && origin !== 'null' && origin !== '*' && [...WEBSITE_ORIGINS, ...configured].includes(origin) ? origin : undefined;
}

/** Apply CORS to the actual response as well as preflight. Request-local
 * headers avoid leaking an allowed origin between concurrent requests. */
export function withCors(handler: (req: Request) => Response | Promise<Response>) {
  return async (req: Request): Promise<Response> => {
    const response = await handler(req);
    const origin = websiteOrigin(req);
    const headers = new Headers(response.headers);
    headers.delete('Access-Control-Allow-Origin');
    if (origin) headers.set('Access-Control-Allow-Origin', origin);
    const vary = headers.get('Vary');
    if (!vary?.split(',').some((part) => part.trim().toLowerCase() === 'origin')) {
      headers.set('Vary', vary ? `${vary}, Origin` : 'Origin');
    }
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  };
}
