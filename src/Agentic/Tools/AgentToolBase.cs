using AiDotNet.Agentic.Models;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Base class for tools that implements the common metadata plumbing, leaving only the behavior
/// (<see cref="InvokeCoreAsync"/>) for subclasses.
/// </summary>
/// <remarks>
/// <para>
/// Follows the template-method pattern used across AiDotNet: this base validates and stores the name,
/// description, and parameter schema, and builds the <see cref="AiToolDefinition"/>; subclasses supply
/// only the execution logic. <see cref="InvokeAsync"/> wraps <see cref="InvokeCoreAsync"/> so unexpected
/// exceptions become a failed <see cref="ToolInvocationResult"/> instead of crashing the agent loop.
/// </para>
/// <para><b>For Beginners:</b> Inherit from this to make a custom tool. You provide the name, the
/// description, the input schema, and one method that does the work. The base class handles the rest,
/// including turning an accidental crash into a clean "this tool failed" message the model can read.
/// </para>
/// </remarks>
public abstract class AgentToolBase : IAgentTool
{
    /// <summary>
    /// Initializes the tool's metadata.
    /// </summary>
    /// <param name="name">The unique tool name.</param>
    /// <param name="description">A natural-language description of what the tool does.</param>
    /// <param name="parametersSchema">The JSON Schema for the tool's arguments; <c>null</c> means no parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> or <paramref name="description"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is empty/whitespace.</exception>
    protected AgentToolBase(string name, string description, JObject? parametersSchema = null)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(description);
        Name = name;
        Description = description;
        ParametersSchema = parametersSchema ?? new JObject { ["type"] = "object", ["properties"] = new JObject() };
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Description { get; }

    /// <inheritdoc/>
    public JObject ParametersSchema { get; }

    /// <inheritdoc/>
    public AiToolDefinition ToDefinition() => new(Name, Description, ParametersSchema);

    /// <inheritdoc/>
    public async Task<ToolInvocationResult> InvokeAsync(JObject arguments, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(arguments);
        try
        {
            return await InvokeCoreAsync(arguments, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException && ex is not StackOverflowException)
        {
            return ToolInvocationResult.Error($"Tool '{Name}' failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Executes the tool's behavior. Implementations read typed values from <paramref name="arguments"/>
    /// and return a <see cref="ToolInvocationResult"/>.
    /// </summary>
    /// <param name="arguments">The validated, non-null argument object.</param>
    /// <param name="cancellationToken">Token used to cancel the invocation.</param>
    /// <returns>A task producing the tool's result.</returns>
    protected abstract Task<ToolInvocationResult> InvokeCoreAsync(JObject arguments, CancellationToken cancellationToken);
}
