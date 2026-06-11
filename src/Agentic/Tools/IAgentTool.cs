using AiDotNet.Agentic.Models;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// An executable tool the model can call: a name, a description, a JSON-schema for its arguments, and
/// an asynchronous invocation entry point.
/// </summary>
/// <remarks>
/// <para>
/// This is the runnable counterpart to <see cref="AiToolDefinition"/>. The definition is what the model
/// is <em>told</em> about; an <see cref="IAgentTool"/> is what actually <em>runs</em> when the model
/// requests a call. The orchestration loop matches a model-emitted
/// <see cref="ToolCallContent"/> to a tool by <see cref="Name"/>, passes the parsed arguments to
/// <see cref="InvokeAsync"/>, and feeds the <see cref="ToolInvocationResult"/> back as a tool message.
/// </para>
/// <para><b>For Beginners:</b> Think of this as one gadget in the assistant's toolbox. It knows its own
/// name, what it does, and what inputs it expects (the schema). When the model says "use this gadget
/// with these inputs", <see cref="InvokeAsync"/> is the code that runs.
/// </para>
/// </remarks>
public interface IAgentTool
{
    /// <summary>
    /// Gets the unique name the model references when requesting this tool.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the natural-language description that helps the model decide when to call the tool.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Gets the JSON Schema object describing this tool's parameters.
    /// </summary>
    JObject ParametersSchema { get; }

    /// <summary>
    /// Produces the provider-facing <see cref="AiToolDefinition"/> for this tool.
    /// </summary>
    /// <returns>A definition that can be placed into <see cref="ChatOptions.Tools"/>.</returns>
    AiToolDefinition ToDefinition();

    /// <summary>
    /// Executes the tool with the supplied arguments.
    /// </summary>
    /// <param name="arguments">The arguments parsed from the model's tool-call JSON. Never <c>null</c>.</param>
    /// <param name="cancellationToken">Token used to cancel the invocation.</param>
    /// <returns>A task producing the tool's result.</returns>
    Task<ToolInvocationResult> InvokeAsync(JObject arguments, CancellationToken cancellationToken = default);
}
