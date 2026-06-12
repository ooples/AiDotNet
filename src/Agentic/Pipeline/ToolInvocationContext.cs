using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// The mutable state flowing through a tool-invocation middleware pipeline: which tool is being called, the
/// (rewritable) arguments, and a shared property bag.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The request slip for running a tool. Middleware can read which tool was asked
/// for and with what inputs, tweak the inputs, or stash notes — before the tool actually runs.
/// </para>
/// </remarks>
public sealed class ToolInvocationContext
{
    /// <summary>
    /// Initializes a new tool-invocation context.
    /// </summary>
    /// <param name="toolName">The name of the tool being invoked.</param>
    /// <param name="arguments">The parsed arguments (middleware may rewrite them). Must be non-null.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="toolName"/> or <paramref name="arguments"/> is <c>null</c>.</exception>
    public ToolInvocationContext(string toolName, JObject arguments)
    {
        Guard.NotNull(toolName);
        Guard.NotNull(arguments);
        ToolName = toolName;
        Arguments = arguments;
    }

    /// <summary>Gets the name of the tool being invoked.</summary>
    public string ToolName { get; }

    /// <summary>Gets or sets the tool arguments (middleware may rewrite them before execution).</summary>
    public JObject Arguments { get; set; }

    /// <summary>Gets a property bag for sharing state between middleware stages.</summary>
    public IDictionary<string, object?> Items { get; } = new Dictionary<string, object?>(StringComparer.Ordinal);
}
