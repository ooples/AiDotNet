using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// A Model Context Protocol (MCP) server that exposes a <see cref="ToolCollection"/> of AiDotNet tools to any
/// MCP client (Claude Desktop, other agent frameworks, or AiDotNet's own <see cref="McpClient"/>). It handles
/// the MCP JSON-RPC methods (<c>initialize</c>, <c>tools/list</c>, <c>tools/call</c>) against the registered
/// tools — turning your tools (including model-as-tool and RAG pipelines) into a standard, interoperable
/// service.
/// </summary>
/// <remarks>
/// <para>
/// This is the inverse of <see cref="McpClient"/>: the client consumes external MCP tools, the server publishes
/// AiDotNet tools. It is transport-agnostic — <see cref="HandleRequestAsync"/> processes a parsed JSON-RPC
/// request, so a stdio/HTTP host (or the in-process <see cref="InMemoryMcpTransport"/>) can drive it.
/// </para>
/// <para><b>For Beginners:</b> The flip side of the client. Instead of using someone else's tools, this lets
/// other AI apps use <em>your</em> AiDotNet tools through the same standard protocol — define a tool once and
/// any MCP-aware client can call it.
/// </para>
/// </remarks>
public sealed class McpServer
{
    private readonly ToolCollection _tools;
    private readonly string _serverName;
    private readonly string _protocolVersion;

    /// <summary>
    /// Initializes a new MCP server over the given tools.
    /// </summary>
    /// <param name="tools">The tools to expose.</param>
    /// <param name="serverName">The server name reported in <c>initialize</c>.</param>
    /// <param name="protocolVersion">The MCP protocol version reported in <c>initialize</c>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="tools"/> is <c>null</c>.</exception>
    public McpServer(ToolCollection tools, string serverName = "AiDotNet", string protocolVersion = "2024-11-05")
    {
        Guard.NotNull(tools);
        Guard.NotNullOrWhiteSpace(serverName);
        Guard.NotNullOrWhiteSpace(protocolVersion);
        _tools = tools;
        _serverName = serverName;
        _protocolVersion = protocolVersion;
    }

    /// <summary>
    /// Handles one MCP JSON-RPC request and returns its <c>result</c> object.
    /// </summary>
    /// <param name="method">The MCP method.</param>
    /// <param name="parameters">The request parameters, or <c>null</c>.</param>
    /// <param name="cancellationToken">Token used to cancel the call.</param>
    /// <returns>The JSON-RPC result object.</returns>
    /// <exception cref="McpException">Thrown for an unknown method or malformed <c>tools/call</c> parameters.</exception>
    public async Task<JObject> HandleRequestAsync(string method, JObject? parameters, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(method);

        switch (method)
        {
            case "initialize":
                return new JObject
                {
                    ["protocolVersion"] = _protocolVersion,
                    ["serverInfo"] = new JObject { ["name"] = _serverName, ["version"] = "1.0" },
                    ["capabilities"] = new JObject { ["tools"] = new JObject() },
                };

            case "tools/list":
                return new JObject { ["tools"] = ToolsArray() };

            case "tools/call":
                return await CallToolAsync(parameters, cancellationToken).ConfigureAwait(false);

            default:
                throw new McpException($"Unsupported MCP method '{method}'.");
        }
    }

    private JArray ToolsArray()
    {
        var array = new JArray();
        foreach (var definition in _tools.GetDefinitions())
        {
            array.Add(new JObject
            {
                ["name"] = definition.Name,
                ["description"] = definition.Description,
                ["inputSchema"] = definition.ParametersSchema,
            });
        }

        return array;
    }

    private async Task<JObject> CallToolAsync(JObject? parameters, CancellationToken cancellationToken)
    {
        var name = (string?)parameters?["name"];
        if (name is null || name.Trim().Length == 0)
        {
            throw new McpException("MCP 'tools/call' requires a non-empty 'name'.");
        }

        // Only a missing 'arguments' may default to {} — an array/string/number
        // there is a malformed request, and coercing it to an empty bag would
        // run the tool with default inputs the caller never asked for.
        var argumentsToken = parameters?["arguments"];
        var arguments = argumentsToken switch
        {
            null => new JObject(),
            JObject argumentsObject => argumentsObject,
            _ => throw new McpException(
                $"MCP 'tools/call' requires 'arguments' to be an object; got {argumentsToken.Type}."),
        };
        var call = new ToolCallContent("mcp-call", name, arguments.ToString(Newtonsoft.Json.Formatting.None));
        var result = await _tools.InvokeAsync(call, cancellationToken).ConfigureAwait(false);

        return new JObject
        {
            ["content"] = new JArray { new JObject { ["type"] = "text", ["text"] = result.Content } },
            ["isError"] = result.IsError,
        };
    }
}
