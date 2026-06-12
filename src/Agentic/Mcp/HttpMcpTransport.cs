using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// An <see cref="IMcpTransport"/> that speaks JSON-RPC 2.0 to a remote MCP server over HTTP POST. Each request
/// is sent as a JSON-RPC envelope and the server's <c>result</c> is returned (a JSON-RPC <c>error</c> becomes
/// an <see cref="McpException"/>).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The way to reach an MCP server that lives on the network. It wraps each call in
/// the standard JSON-RPC envelope, posts it, and unwraps the answer — so <see cref="McpClient"/> works against
/// a hosted server exactly as it does against an in-process one.
/// </para>
/// </remarks>
public sealed class HttpMcpTransport : IMcpTransport, IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _endpoint;
    private readonly bool _ownsHttpClient;
    private int _id;

    /// <summary>
    /// Initializes a new HTTP transport.
    /// </summary>
    /// <param name="endpoint">The MCP server URL.</param>
    /// <param name="httpClient">
    /// Optional HTTP client (for testing or custom handlers/headers). When supplied, the caller retains
    /// ownership and must dispose it; when omitted, the transport creates and owns its own client.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="endpoint"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="endpoint"/> is empty/whitespace.</exception>
    public HttpMcpTransport(string endpoint, HttpClient? httpClient = null)
    {
        Guard.NotNullOrWhiteSpace(endpoint);
        _endpoint = endpoint;
        _httpClient = httpClient ?? new HttpClient();
        // Only dispose what we created: a caller-supplied client may be shared
        // with other transports/components.
        _ownsHttpClient = httpClient is null;
    }

    /// <summary>Disposes the internally created <see cref="HttpClient"/> (caller-supplied clients are left alone).</summary>
    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }

    /// <inheritdoc/>
    public async Task<JObject> SendRequestAsync(string method, JObject? parameters, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(method);

        var id = Interlocked.Increment(ref _id);
        var envelope = McpJsonRpc.BuildRequest(id, method, parameters);

        using var request = new HttpRequestMessage(HttpMethod.Post, _endpoint)
        {
            Content = new StringContent(envelope.ToString(Formatting.None), Encoding.UTF8, "application/json"),
        };
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        return McpJsonRpc.ParseResult(body);
    }
}
