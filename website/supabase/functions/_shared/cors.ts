const WEBSITE_ORIGINS = ['https://www.aidotnet.dev', 'https://test.aidotnet.dev'];

/** Apply CORS to the actual response as well as preflight. Request-local
 * headers avoid leaking an allowed origin between concurrent requests. */
export function withCors(handler: (req: Request) => Response | Promise<Response>) {
  return async (req: Request): Promise<Response> => {
    const response = await handler(req);
    const configured = [Deno.env.get('ALLOWED_ORIGIN'), Deno.env.get('ALLOWED_ORIGINS')]
      .filter((value): value is string => !!value).flatMap((value) => value.split(',').map((origin) => origin.trim()));
    const allowed = new Set([...WEBSITE_ORIGINS, ...configured]);
    const origin = req.headers.get('Origin');
    const headers = new Headers(response.headers);
    headers.delete('Access-Control-Allow-Origin');
    if (origin && allowed.has(origin) && origin !== 'null' && origin !== '*') headers.set('Access-Control-Allow-Origin', origin);
    const vary = headers.get('Vary');
    if (!vary?.split(',').some((part) => part.trim().toLowerCase() === 'origin')) {
      headers.set('Vary', vary ? `${vary}, Origin` : 'Origin');
    }
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  };
}
