namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Adds a description (and optional required-override) to a tool method parameter, enriching the
/// auto-generated JSON schema.
/// </summary>
/// <remarks>
/// <para>
/// Parameter descriptions materially improve how well a model fills in tool arguments. By default a
/// parameter is required unless it has a default value or is nullable; set <see cref="Required"/> to
/// override that inference.
/// </para>
/// <para><b>For Beginners:</b> Put this on a tool method's parameter to tell the model what that input
/// means — e.g. <c>[ToolParameter("City name, e.g. 'Paris'")] string city</c>. Clearer descriptions
/// lead to more accurate tool calls.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = true)]
public sealed class ToolParameterAttribute : Attribute
{
    /// <summary>
    /// Initializes the attribute.
    /// </summary>
    /// <param name="description">A natural-language description of the parameter.</param>
    public ToolParameterAttribute(string description = "")
    {
        Description = description ?? string.Empty;
    }

    /// <summary>
    /// Gets the natural-language description of the parameter.
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Gets or sets an explicit override for whether the parameter is required. When <c>null</c>,
    /// requiredness is inferred from the parameter's default value and nullability.
    /// </summary>
    public bool? Required { get; set; }
}
