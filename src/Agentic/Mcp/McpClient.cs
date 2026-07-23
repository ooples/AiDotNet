using System.Text;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// A Model Context Protocol (MCP) client: connects to an MCP server over an <see cref="IMcpTransport"/>,
/// lists the tools it offers, and exposes them as <see cref="IAgentTool"/> instances the agent stack can call
/// — so any MCP server's capabilities become available to AiDotNet agents with no per-tool code.
/// </summary>
/// <remarks>
/// <para>
/// MCP is the emerging standard (used by Semantic Kernel, LangGraph, and others) for connecting models to
/// external tools/data. <see cref="GetToolsAsync"/> returns a <see cref="ToolCollection"/> of adapters that
/// forward calls to the server, so an MCP-hosted tool is indistinguishable from a native one to the model.
/// </para>
/// <para><b>For Beginners:</b> MCP servers publish tools (search the web, read a database, control an app).
/// This client connects to one, asks "what tools do you have?", and wraps each so your agent can use them
/// exactly like its own — instantly expanding what it can do.
/// </para>
/// </remarks>
public sealed class McpClient
{
    private readonly IMcpTransport _transport;

    /// <summary>
    /// Initializes a new MCP client over the given transport.
    /// </summary>
    /// <param name="transport">The transport to the MCP server.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="transport"/> is <c>null</c>.</exception>
    public McpClient(IMcpTransport transport)
    {
        Guard.NotNull(transport);
        _transport = transport;
    }

    /// <summary>
    /// Performs the MCP <c>initialize</c> handshake and returns the server's capabilities/info object.
    /// </summary>
    /// <param name="protocolVersion">The MCP protocol version the client speaks.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The server's <c>initialize</c> result.</returns>
    public Task<JObject> InitializeAsync(string protocolVersion = "2024-11-05", CancellationToken cancellationToken = default)
    {
        var parameters = new JObject
        {
            ["protocolVersion"] = protocolVersion,
            ["clientInfo"] = new JObject { ["name"] = "AiDotNet", ["version"] = "1.0" },
            ["capabilities"] = new JObject(),
        };
        return _transport.SendRequestAsync("initialize", parameters, cancellationToken);
    }

    /// <summary>
    /// Lists the tools the server offers (the MCP <c>tools/list</c> method).
    /// </summary>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The server's tool descriptors.</returns>
    /// <exception cref="McpException">Thrown when the response is malformed.</exception>
    public async Task<IReadOnlyList<McpToolDescriptor>> ListToolsAsync(CancellationToken cancellationToken = default)
    {
        var result = await _transport.SendRequestAsync("tools/list", null, cancellationToken).ConfigureAwait(false);
        if (result["tools"] is not JArray toolsArray)
        {
            throw new McpException("MCP 'tools/list' response did not contain a 'tools' array.");
        }

        var descriptors = new List<McpToolDescriptor>(toolsArray.Count);
        for (var i = 0; i < toolsArray.Count; i++)
        {
            var entry = toolsArray[i];
            var name = (string?)entry["name"];
            // A nameless tool entry is a server contract violation — surface
            // it as the documented McpException instead of silently dropping
            // the tool and leaving the caller with a mysteriously short list.
            if (name is null || name.Trim().Length == 0)
            {
                throw new McpException($"MCP 'tools/list' entry at index {i} is missing a valid 'name'.");
            }

            var description = (string?)entry["description"] ?? string.Empty;
            var schema = entry["inputSchema"] as JObject ?? new JObject { ["type"] = "object", ["properties"] = new JObject() };
            descriptors.Add(new McpToolDescriptor(name, description, schema));
        }

        return descriptors;
    }

    /// <summary>
    /// Calls a tool on the server (the MCP <c>tools/call</c> method) and returns its result as a
    /// <see cref="ToolInvocationResult"/>.
    /// </summary>
    /// <param name="name">The tool name.</param>
    /// <param name="arguments">The tool arguments.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The tool result (error when the server reports <c>isError</c>).</returns>
    public async Task<ToolInvocationResult> CallToolAsync(string name, JObject arguments, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(arguments);

        var parameters = new JObject { ["name"] = name, ["arguments"] = arguments };
        var result = await _transport.SendRequestAsync("tools/call", parameters, cancellationToken).ConfigureAwait(false);

        var text = ExtractText(result["content"]);
        var isError = (bool?)result["isError"] ?? false;
        return isError ? ToolInvocationResult.Error(text) : ToolInvocationResult.Success(text);
    }

    /// <summary>
    /// Lists the server's tools and wraps each as an <see cref="IAgentTool"/> in a ready-to-use
    /// <see cref="ToolCollection"/>.
    /// </summary>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>A collection of MCP-backed tools.</returns>
    public async Task<ToolCollection> GetToolsAsync(CancellationToken cancellationToken = default)
    {
        var descriptors = await ListToolsAsync(cancellationToken).ConfigureAwait(false);
        var collection = new ToolCollection();
        foreach (var descriptor in descriptors)
        {
            collection.Add(new McpToolAdapter(this, descriptor));
        }

        return collection;
    }

    private static string ExtractText(JToken? content)
    {
        if (content is not JArray parts)
        {
            return content is null ? string.Empty : content.ToString();
        }

        var builder = new StringBuilder();
        foreach (var part in parts)
        {
            var text = (string?)part["text"];
            if (text is not null)
            {
                builder.Append(text);
            }
        }

        return builder.ToString();
    }
}
