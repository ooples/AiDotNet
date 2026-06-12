using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// Describes a tool advertised by an MCP server: its name, description, and JSON-Schema input contract — the
/// data needed to surface it to a model as a callable tool.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> One entry from an MCP server's tool menu. <see cref="McpClient"/> turns each of
/// these into a tool your agent can call, no different from a built-in one.
/// </para>
/// </remarks>
public sealed class McpToolDescriptor
{
    private readonly JObject _inputSchema;

    /// <summary>
    /// Initializes a new descriptor.
    /// </summary>
    /// <param name="name">The tool name. Must be non-empty.</param>
    /// <param name="description">The tool description.</param>
    /// <param name="inputSchema">The JSON-Schema for the tool's arguments.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/>, <paramref name="description"/>, or <paramref name="inputSchema"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is empty/whitespace.</exception>
    public McpToolDescriptor(string name, string description, JObject inputSchema)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(description);
        Guard.NotNull(inputSchema);
        Name = name;
        Description = description;
        // Snapshot the schema: JObject is mutable, and a descriptor must not
        // change after construction just because the caller (or a consumer of
        // InputSchema) edits the instance it handed in / got back.
        _inputSchema = (JObject)inputSchema.DeepClone();
    }

    /// <summary>Gets the tool name.</summary>
    public string Name { get; }

    /// <summary>Gets the tool description.</summary>
    public string Description { get; }

    /// <summary>Gets a copy of the JSON-Schema describing the tool's arguments (mutating it does not affect the descriptor).</summary>
    public JObject InputSchema => (JObject)_inputSchema.DeepClone();
}
