using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// Adapts an MCP server tool (<see cref="McpToolDescriptor"/>) to <see cref="IAgentTool"/>, forwarding
/// invocations to the server via <see cref="McpClient"/>. Register it in a tool collection and the agent loop
/// calls the remote tool exactly as if it were local.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The wrapper that makes a tool living on an MCP server look and behave like one
/// of your agent's own tools — when the model calls it, the call is sent to the server and the answer comes
/// back.
/// </para>
/// </remarks>
public sealed class McpToolAdapter : IAgentTool
{
    private readonly McpClient _client;
    private readonly McpToolDescriptor _descriptor;

    /// <summary>
    /// Initializes a new adapter.
    /// </summary>
    /// <param name="client">The MCP client used to invoke the tool.</param>
    /// <param name="descriptor">The server's description of the tool.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> or <paramref name="descriptor"/> is <c>null</c>.</exception>
    public McpToolAdapter(McpClient client, McpToolDescriptor descriptor)
    {
        Guard.NotNull(client);
        Guard.NotNull(descriptor);
        _client = client;
        _descriptor = descriptor;
    }

    /// <inheritdoc/>
    public string Name => _descriptor.Name;

    /// <inheritdoc/>
    public string Description => _descriptor.Description;

    /// <inheritdoc/>
    public JObject ParametersSchema => _descriptor.InputSchema;

    /// <inheritdoc/>
    public AiToolDefinition ToDefinition() => new(Name, Description, ParametersSchema);

    /// <inheritdoc/>
    public Task<ToolInvocationResult> InvokeAsync(JObject arguments, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(arguments);
        return _client.CallToolAsync(Name, arguments, cancellationToken);
    }
}
