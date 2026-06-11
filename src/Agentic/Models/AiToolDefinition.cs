using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Models;

/// <summary>
/// Describes a tool/function the model is allowed to call: its name, a description, and a JSON-schema
/// describing its parameters.
/// </summary>
/// <remarks>
/// <para>
/// This is the <em>declaration</em> sent to the model (distinct from the executable tool in the Tools
/// layer). The model uses the name and description to decide <em>whether</em> to call the tool, and the
/// parameter schema to decide <em>how</em> to fill in the arguments. The schema is a standard JSON Schema
/// object (the same shape OpenAI, Anthropic, and the local engine all consume).
/// </para>
/// <para><b>For Beginners:</b> Before a model can use a tool, you have to describe the tool to it — like
/// a menu entry: the tool's name, what it does, and what inputs it needs. This class is that menu entry.
/// The "what inputs it needs" part is written in JSON Schema (e.g., "an object with a string field
/// called <c>city</c>"). In later phases this schema is generated automatically from your C# method
/// signature, so you won't usually hand-write it.
/// </para>
/// </remarks>
public sealed class AiToolDefinition
{
    /// <summary>
    /// Initializes a new tool definition.
    /// </summary>
    /// <param name="name">The unique tool name the model will reference when calling it.</param>
    /// <param name="description">A natural-language description of what the tool does.</param>
    /// <param name="parametersSchema">
    /// A JSON Schema object describing the tool's parameters. When <c>null</c>, an empty
    /// parameter-less schema (<c>{"type":"object","properties":{}}</c>) is used.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> or <paramref name="description"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is empty/whitespace.</exception>
    public AiToolDefinition(string name, string description, JObject? parametersSchema = null)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(description);
        Name = name;
        Description = description;
        ParametersSchema = parametersSchema ?? CreateEmptySchema();
    }

    /// <summary>
    /// Gets the unique name the model references when requesting this tool.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the natural-language description that helps the model decide when to call the tool.
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Gets the JSON Schema object describing the tool's parameters.
    /// </summary>
    public JObject ParametersSchema { get; }

    private static JObject CreateEmptySchema() =>
        new() { ["type"] = "object", ["properties"] = new JObject() };
}
