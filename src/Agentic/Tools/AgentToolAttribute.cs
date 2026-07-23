namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Marks a method as an agent tool so it can be discovered and exposed to a model with an
/// auto-generated JSON schema.
/// </summary>
/// <remarks>
/// <para>
/// Annotate a public method, then register the containing object with the tool layer; the method's
/// signature is turned into an <see cref="AiDotNet.Agentic.Models.AiToolDefinition"/> automatically.
/// The tool name defaults to the method name when <see cref="Name"/> is not set.
/// </para>
/// <para><b>For Beginners:</b> Instead of hand-writing a tool class, you can write a normal C# method
/// (e.g., <c>GetWeather(string city)</c>), put <c>[AgentTool("Gets current weather")]</c> on it, and
/// the library figures out the tool's name, description, and input schema for you.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = true)]
public sealed class AgentToolAttribute : Attribute
{
    /// <summary>
    /// Initializes the attribute.
    /// </summary>
    /// <param name="description">A natural-language description of what the tool does.</param>
    public AgentToolAttribute(string description = "")
    {
        Description = description ?? string.Empty;
    }

    /// <summary>
    /// Gets or sets the tool name exposed to the model. When <c>null</c> or empty, the method name is used.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Gets the natural-language description of the tool.
    /// </summary>
    public string Description { get; }
}
