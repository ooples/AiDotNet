using System.Text;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// Adapts any <see cref="IAgent{T}"/> into an <see cref="IAgentTool"/>, so one agent can delegate a
/// sub-task to another by calling it as a native tool (a "handoff"). This is the building block of the
/// supervisor pattern: each worker agent becomes a <c>transfer_to_&lt;worker&gt;</c> tool the coordinator
/// can invoke.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// The wrapped agent is exposed with a single required <c>task</c> string parameter. When the coordinator
/// calls the tool, the supplied task is run through the worker as a fresh user message and the worker's
/// final answer is returned as the tool result, which the coordinator then sees on its next turn.
/// </para>
/// <para><b>For Beginners:</b> This turns a teammate into a "button" the lead agent can press. The button
/// is labelled <c>transfer_to_&lt;teammate&gt;</c>; pressing it hands the teammate a task and gives back
/// their answer. Because a team member is just another tool, the coordinator uses the exact same
/// tool-calling machinery it would for a calculator or a web search.
/// </para>
/// </remarks>
internal sealed class AgentAsTool<T> : IAgentTool
{
    /// <summary>The conventional prefix applied to a wrapped agent's tool name.</summary>
    public const string DefaultNamePrefix = "transfer_to_";

    private readonly IAgent<T> _agent;

    /// <summary>
    /// Initializes a new adapter exposing <paramref name="agent"/> as a callable tool.
    /// </summary>
    /// <param name="agent">The agent to expose. Its <see cref="IAgent{T}.Name"/> is sanitized into the tool name.</param>
    /// <param name="namePrefix">The tool-name prefix. <c>null</c> uses <see cref="DefaultNamePrefix"/>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="agent"/> is <c>null</c>.</exception>
    public AgentAsTool(IAgent<T> agent, string? namePrefix = null)
    {
        Guard.NotNull(agent);
        _agent = agent;
        var prefix = namePrefix ?? DefaultNamePrefix;
        // Sanitize the COMPOSED name, not just the agent part: a caller-supplied
        // prefix like "transfer to " carries provider-invalid characters that
        // downstream chat/tool APIs reject.
        Name = ToolNaming.Sanitize(prefix + _agent.Name);
        Description = BuildDescription(_agent);
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Description { get; }

    // Shared parameter schema — identical for every AgentAsTool<T> handoff,
    // so we keep one instance per process instead of allocating a fresh
    // JObject per tool. (Note: callers must not mutate the returned object;
    // ToDefinition just hands it to AiToolDefinition which treats it as
    // opaque metadata.)
    private static readonly JObject SharedParametersSchema = new()
    {
        ["type"] = "object",
        ["properties"] = new JObject
        {
            ["task"] = new JObject
            {
                ["type"] = "string",
                ["description"] = "The self-contained task or question to delegate to this agent."
            }
        },
        ["required"] = new JArray("task")
    };

    /// <inheritdoc/>
    public JObject ParametersSchema => SharedParametersSchema;

    /// <inheritdoc/>
    public AiToolDefinition ToDefinition() => new(Name, Description, ParametersSchema);

    /// <inheritdoc/>
    public async Task<ToolInvocationResult> InvokeAsync(JObject arguments, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(arguments);
        // Validate the token shape before reading — JObject's implicit
        // cast to string returns null for any non-string token type and
        // would otherwise mask "the model passed an object/number/null"
        // as a generic missing-argument error.
        if (!arguments.TryGetValue("task", out var taskToken)
            || taskToken is null
            || taskToken.Type == JTokenType.Null
            || taskToken.Type != JTokenType.String)
        {
            return ToolInvocationResult.Error(
                $"The '{Name}' handoff requires a non-empty 'task' string argument.");
        }
        var task = taskToken.Value<string>();
        if (task is null || task.Trim().Length == 0)
        {
            return ToolInvocationResult.Error(
                $"The '{Name}' handoff requires a non-empty 'task' string argument.");
        }

        var result = await _agent.RunAsync(new[] { ChatMessage.User(task) }, cancellationToken)
            .ConfigureAwait(false);

        if (!result.Completed)
        {
            return ToolInvocationResult.Error(
                $"Agent '{_agent.Name}' did not finish the task within its step budget. Partial output: {result.FinalText}");
        }

        return ToolInvocationResult.Success(result.FinalText);
    }

    private static string BuildDescription(IAgent<T> agent)
    {
        var builder = new StringBuilder();
        builder.Append("Delegate a self-contained task to the '").Append(agent.Name).Append("' agent.");
        if (agent.Description.Trim().Length > 0)
        {
            builder.Append(' ').Append(agent.Description.Trim());
        }

        return builder.ToString();
    }
}
