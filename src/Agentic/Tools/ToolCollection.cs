using AiDotNet.Agentic.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage, which is in scope
// project-wide via a global using in AiModelBuilder.cs. The new agentic subsystem uses the Models type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// A named set of executable tools: registers <see cref="IAgentTool"/> instances, exposes their
/// <see cref="AiToolDefinition"/>s for a chat request, and dispatches model tool-calls to the right tool.
/// </summary>
/// <remarks>
/// <para>
/// This is the bridge between the model and your code during a tool-calling turn. Register tools, hand
/// <see cref="GetDefinitions"/> to <see cref="ChatOptions.Tools"/>, and when the model replies with
/// <see cref="ToolCallContent"/>, call <see cref="InvokeToToolMessageAsync"/> to run the tool and produce
/// the <see cref="ChatRole.Tool"/> message to append before the next model call.
/// </para>
/// <para><b>For Beginners:</b> Think of this as the labeled toolbox you give the model. It can list what's
/// inside (so the model knows its options) and, when the model asks to use a specific tool, it finds that
/// tool, runs it, and packages the answer to hand back to the model.
/// </para>
/// </remarks>
public sealed class ToolCollection
{
    private readonly Dictionary<string, IAgentTool> _tools = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the number of registered tools.
    /// </summary>
    public int Count => _tools.Count;

    /// <summary>
    /// Gets the registered tools.
    /// </summary>
    public IReadOnlyList<IAgentTool> Tools => _tools.Values.ToList();

    /// <summary>
    /// Registers a tool.
    /// </summary>
    /// <param name="tool">The tool to add.</param>
    /// <returns>This collection, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="tool"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when a tool with the same name is already registered.</exception>
    public ToolCollection Add(IAgentTool tool)
    {
        Guard.NotNull(tool);
        if (_tools.ContainsKey(tool.Name))
        {
            throw new ArgumentException($"A tool named '{tool.Name}' is already registered.", nameof(tool));
        }

        _tools[tool.Name] = tool;
        return this;
    }

    /// <summary>
    /// Creates a tool from a delegate and registers it.
    /// </summary>
    /// <param name="name">The tool name.</param>
    /// <param name="description">A description of what the tool does.</param>
    /// <param name="function">The delegate to invoke.</param>
    /// <returns>This collection, for chaining.</returns>
    public ToolCollection AddDelegate(string name, string description, Delegate function) =>
        Add(AgentToolFactory.FromDelegate(name, description, function));

    /// <summary>
    /// Scans an object for <see cref="AgentToolAttribute"/>-annotated methods and registers each as a tool.
    /// </summary>
    /// <param name="target">The object to scan.</param>
    /// <returns>This collection, for chaining.</returns>
    public ToolCollection AddFrom(object target)
    {
        foreach (var tool in AgentToolFactory.ScanInstance(target))
        {
            Add(tool);
        }

        return this;
    }

    /// <summary>
    /// Determines whether a tool with the given name is registered.
    /// </summary>
    /// <param name="name">The tool name (case-insensitive).</param>
    /// <returns><c>true</c> when present; otherwise <c>false</c>.</returns>
    public bool Contains(string name) => name is not null && _tools.ContainsKey(name);

    /// <summary>
    /// Gets a tool by name.
    /// </summary>
    /// <param name="name">The tool name (case-insensitive).</param>
    /// <returns>The tool, or <c>null</c> when not found.</returns>
    public IAgentTool? Get(string name) =>
        name is not null && _tools.TryGetValue(name, out var tool) ? tool : null;

    /// <summary>
    /// Produces the provider-facing definitions for all registered tools, for use as
    /// <see cref="ChatOptions.Tools"/>.
    /// </summary>
    /// <returns>The tool definitions.</returns>
    public IReadOnlyList<AiToolDefinition> GetDefinitions() =>
        _tools.Values.Select(t => t.ToDefinition()).ToList();

    /// <summary>
    /// Executes a model-emitted tool call and returns the raw result.
    /// </summary>
    /// <param name="call">The tool call requested by the model.</param>
    /// <param name="cancellationToken">Token used to cancel the invocation.</param>
    /// <returns>The tool's result, or an error result when the tool is unknown or the arguments are malformed.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="call"/> is <c>null</c>.</exception>
    public async Task<ToolInvocationResult> InvokeAsync(ToolCallContent call, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(call);

        var tool = Get(call.ToolName);
        if (tool is null)
        {
            return ToolInvocationResult.Error(
                $"Unknown tool '{call.ToolName}'. Available tools: {string.Join(", ", _tools.Keys)}");
        }

        JObject arguments;
        try
        {
            arguments = string.IsNullOrWhiteSpace(call.ArgumentsJson)
                ? new JObject()
                : JObject.Parse(call.ArgumentsJson);
        }
        catch (JsonException ex)
        {
            return ToolInvocationResult.Error($"Arguments for tool '{call.ToolName}' were not valid JSON: {ex.Message}");
        }

        return await tool.InvokeAsync(arguments, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Executes a model-emitted tool call and wraps the result in a <see cref="ChatRole.Tool"/> message
    /// ready to append to the conversation before the next model call.
    /// </summary>
    /// <param name="call">The tool call requested by the model.</param>
    /// <param name="cancellationToken">Token used to cancel the invocation.</param>
    /// <returns>A tool-role <see cref="ChatMessage"/> carrying the result.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="call"/> is <c>null</c>.</exception>
    public async Task<ChatMessage> InvokeToToolMessageAsync(ToolCallContent call, CancellationToken cancellationToken = default)
    {
        var result = await InvokeAsync(call, cancellationToken).ConfigureAwait(false);
        return ChatMessage.Tool(call.CallId, result.Content, result.IsError);
    }
}
