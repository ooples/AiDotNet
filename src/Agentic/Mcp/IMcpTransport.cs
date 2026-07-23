using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// Carries Model Context Protocol (MCP) JSON-RPC requests to a server and returns the result. Abstracting the
/// transport keeps <see cref="McpClient"/> independent of how messages are framed — stdio for local server
/// processes, HTTP/SSE for remote servers, or an in-memory loopback for tests.
/// </summary>
/// <remarks>
/// <para>
/// MCP is JSON-RPC 2.0. An implementation sends a request for <paramref name="method"/> with
/// <paramref name="parameters"/> and returns the JSON-RPC <c>result</c> object; on a JSON-RPC error it should
/// throw an <see cref="McpException"/>.
/// </para>
/// <para><b>For Beginners:</b> The pipe to an MCP server. The client says "call this method with these
/// arguments" and the pipe delivers it and brings back the answer — whether the server is a local program or a
/// remote service.
/// </para>
/// </remarks>
public interface IMcpTransport
{
    /// <summary>
    /// Sends a JSON-RPC request and returns its <c>result</c> object.
    /// </summary>
    /// <param name="method">The MCP method (e.g., <c>initialize</c>, <c>tools/list</c>, <c>tools/call</c>).</param>
    /// <param name="parameters">The request parameters, or <c>null</c>.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The JSON-RPC result object.</returns>
    Task<JObject> SendRequestAsync(string method, JObject? parameters, CancellationToken cancellationToken = default);
}

/// <summary>
/// Thrown when an MCP server returns a JSON-RPC error or a malformed response.
/// </summary>
public sealed class McpException : Exception
{
    /// <summary>Initializes a new MCP exception.</summary>
    /// <param name="message">The error message.</param>
    public McpException(string message)
        : base(message)
    {
    }
}
