using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// An <see cref="IAgentTool"/> whose parameter schema is supplied up front and whose execution is a
/// caller-provided delegate — i.e. no reflection at invoke time. This is the target type for the
/// source generator, which emits a precomputed schema and a typed argument binder for each
/// <see cref="AgentToolAttribute"/>-annotated method.
/// </summary>
/// <remarks>
/// <para>
/// Compared to <see cref="DelegateAgentTool"/> (which inspects a method via reflection at construction
/// and invocation), this type carries an already-built schema and a ready-to-run invoker, making it
/// AOT-friendly and allocation-light. You rarely construct it by hand; the generated
/// <c>CreateAgentTools()</c> extension produces these for you.
/// </para>
/// <para><b>For Beginners:</b> This is a tool whose "what inputs do I take" (schema) and "what do I do"
/// (the invoker) are both handed in ready-made, so nothing has to be figured out by reflection while the
/// agent runs.
/// </para>
/// </remarks>
public sealed class FunctionAgentTool : AgentToolBase
{
    private readonly Func<JObject, CancellationToken, Task<string>> _invoker;

    /// <summary>
    /// Initializes a new function-backed tool.
    /// </summary>
    /// <param name="name">The tool name exposed to the model.</param>
    /// <param name="description">A description of what the tool does.</param>
    /// <param name="parametersSchema">The precomputed JSON Schema for the tool's parameters.</param>
    /// <param name="invoker">The delegate that binds the arguments and runs the tool, returning its text result.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="invoker"/> is <c>null</c> (other nulls are validated by the base).</exception>
    public FunctionAgentTool(
        string name,
        string description,
        JObject parametersSchema,
        Func<JObject, CancellationToken, Task<string>> invoker)
        : base(name, description, parametersSchema)
    {
        Guard.NotNull(invoker);
        _invoker = invoker;
    }

    /// <inheritdoc/>
    protected override async Task<ToolInvocationResult> InvokeCoreAsync(JObject arguments, CancellationToken cancellationToken)
    {
        var content = await _invoker(arguments, cancellationToken).ConfigureAwait(false);
        return ToolInvocationResult.Success(content);
    }
}
