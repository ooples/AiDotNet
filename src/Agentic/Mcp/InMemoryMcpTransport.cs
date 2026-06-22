using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// An <see cref="IMcpTransport"/> that forwards requests directly to an in-process <see cref="McpServer"/> —
/// no serialization or network. Useful for tests, for embedding an MCP server in the same process as its
/// client, and for composing AiDotNet tools as MCP tools without a transport hop.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A short-circuit pipe: instead of sending MCP messages over stdio or HTTP, it
/// hands them straight to a server object running in the same program. Great for wiring a client to a server
/// in one process (and for testing the two together).
/// </para>
/// </remarks>
public sealed class InMemoryMcpTransport : IMcpTransport
{
    private readonly McpServer _server;

    /// <summary>
    /// Initializes a new in-memory transport over the given server.
    /// </summary>
    /// <param name="server">The server that handles requests.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="server"/> is <c>null</c>.</exception>
    public InMemoryMcpTransport(McpServer server)
    {
        Guard.NotNull(server);
        _server = server;
    }

    /// <inheritdoc/>
    public Task<JObject> SendRequestAsync(string method, JObject? parameters, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(method);
        return _server.HandleRequestAsync(method, parameters, cancellationToken);
    }
}
