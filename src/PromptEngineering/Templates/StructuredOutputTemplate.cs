using System.Text;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Template that guides models to produce structured output in specific formats.
/// </summary>
/// <remarks>
/// <para>
/// This template helps ensure consistent, parseable output by providing
/// format specifications, schemas, and examples of the expected structure.
/// </para>
/// <para><b>For Beginners:</b> Gets AI to output data in a specific format.
///
/// Example:
/// <code>
/// var template = new StructuredOutputTemplate(
///     OutputFormat.Json,
///     schema: @"{
///         ""name"": ""string"",
///         ""age"": ""number"",
///         ""email"": ""string""
///     }"
/// );
///
/// var prompt = template.WithTask("Extract user info from: John Doe, 30, john@email.com")
///     .Format(new Dictionary&lt;string, string&gt;());
/// </code>
///
/// Supported formats:
/// - JSON for APIs and data processing
/// - XML for configuration and interchange
/// - CSV for tabular data
/// - Markdown for documentation
/// - Custom formats with user-defined schemas
/// </para>
/// </remarks>
public class StructuredOutputTemplate : PromptTemplateBase
{
    private readonly OutputFormat _format;
    private readonly string? _schema;
    private readonly string? _example;

    /// <summary>
    /// Supported output formats.
    /// </summary>
    public enum OutputFormat
    {
        /// <summary>JSON format for structured data.</summary>
        Json,

        /// <summary>XML format for structured data.</summary>
        Xml,

        /// <summary>CSV format for tabular data.</summary>
        Csv,

        /// <summary>Markdown format for documentation.</summary>
        Markdown,

        /// <summary>YAML format for configuration.</summary>
        Yaml,

        /// <summary>Plain text with custom structure.</summary>
        Custom
    }

    /// <summary>
    /// Initializes a new instance of the StructuredOutputTemplate class.
    /// </summary>
    /// <param name="format">The desired output format.</param>
    /// <param name="schema">Optional schema definition for the output.</param>
    /// <param name="example">Optional example of the expected output.</param>
    public StructuredOutputTemplate(OutputFormat format, string? schema = null, string? example = null)
        : base(BuildTemplate(format, schema, example, null))
    {
        _format = format;
        _schema = schema;
        _example = example;
    }

    /// <summary>
    /// Initializes a new instance with a custom template string.
    /// </summary>
    /// <param name="template">The custom template string.</param>
    public StructuredOutputTemplate(string template)
        : base(template)
    {
        _format = OutputFormat.Custom;
    }

    private static string BuildTemplate(OutputFormat format, string? schema, string? example, string? task)
    {
        var sb = new StringBuilder();

        // Add task if provided
        if (!string.IsNullOrWhiteSpace(task))
        {
            sb.AppendLine($"Task: {task}");
            sb.AppendLine();
        }
        else
        {
            sb.AppendLine("Task: {task}");
            sb.AppendLine();
        }

        // Add format instructions
        sb.AppendLine($"Output Format: {GetFormatName(format)}");
        sb.AppendLine();

        // Add format-specific instructions
        sb.AppendLine(GetFormatInstructions(format));
        sb.AppendLine();

        // Add schema if provided
        if (!string.IsNullOrWhiteSpace(schema))
        {
            sb.AppendLine("Schema:");
            sb.AppendLine("```");
            sb.AppendLine(schema);
            sb.AppendLine("```");
            sb.AppendLine();
        }

        // Add example if provided
        if (!string.IsNullOrWhiteSpace(example))
        {
            sb.AppendLine("Example output:");
            sb.AppendLine("```");
            sb.AppendLine(example);
            sb.AppendLine("```");
            sb.AppendLine();
        }

        sb.AppendLine("Important:");
        sb.AppendLine("- Output ONLY the formatted data, no explanations");
        sb.AppendLine("- Follow the schema/format exactly");
        sb.AppendLine("- Ensure the output is valid and parseable");
        sb.AppendLine();
        sb.AppendLine("Your structured output:");

        return sb.ToString();
    }

    private static string GetFormatName(OutputFormat format)
    {
        return format switch
        {
            OutputFormat.Json => "JSON",
            OutputFormat.Xml => "XML",
            OutputFormat.Csv => "CSV",
            OutputFormat.Markdown => "Markdown",
            OutputFormat.Yaml => "YAML",
            OutputFormat.Custom => "Custom",
            _ => "Text"
        };
    }

    private static string GetFormatInstructions(OutputFormat format)
    {
        return format switch
        {
            OutputFormat.Json => @"Requirements:
- Use valid JSON syntax
- Use double quotes for strings
- Use proper data types (strings, numbers, booleans, arrays, objects)
- Ensure proper nesting and formatting",

            OutputFormat.Xml => @"Requirements:
- Use valid XML syntax
- Include proper opening and closing tags
- Escape special characters (&, <, >, "", ')
- Use consistent indentation",

            OutputFormat.Csv => @"Requirements:
- Use commas as field separators
- Include a header row
- Quote fields containing commas or quotes
- Use consistent row structure",

            OutputFormat.Markdown => @"Requirements:
- Use proper Markdown syntax
- Use headers (#, ##, ###) for sections
- Use lists (-, *, 1.) where appropriate
- Use code blocks for code snippets",

            OutputFormat.Yaml => @"Requirements:
- Use valid YAML syntax
- Use proper indentation (2 spaces)
- Quote strings when necessary
- Use proper data types",

            _ => @"Requirements:
- Follow the specified format exactly
- Ensure consistency throughout the output"
        };
    }

    /// <summary>
    /// Sets the task to perform.
    /// </summary>
    /// <param name="task">The task description.</param>
    /// <returns>A new template with the task set.</returns>
    public StructuredOutputTemplate WithTask(string task)
    {
        return new StructuredOutputTemplate(BuildTemplate(_format, _schema, _example, task));
    }

    /// <summary>
    /// Formats the structured output template.
    /// </summary>
    protected override string FormatCore(Dictionary<string, string> variables)
    {
        var result = Template;

        foreach (var kvp in variables)
        {
            var placeholder = $"{{{kvp.Key}}}";
            result = result.Replace(placeholder, kvp.Value ?? string.Empty);
        }

        return result.Trim();
    }

    /// <summary>
    /// Creates a JSON output template with a schema.
    /// </summary>
    /// <param name="schema">The JSON schema.</param>
    /// <param name="example">Optional example output.</param>
    public static StructuredOutputTemplate Json(string? schema = null, string? example = null)
    {
        return new StructuredOutputTemplate(OutputFormat.Json, schema, example);
    }

    /// <summary>
    /// Creates an XML output template with a schema.
    /// </summary>
    /// <param name="rootElement">The root element name.</param>
    /// <param name="schema">Optional schema definition.</param>
    public static StructuredOutputTemplate Xml(string rootElement, string? schema = null)
    {
        var defaultSchema = schema ?? $"<{rootElement}>\n  <!-- Your content here -->\n</{rootElement}>";
        return new StructuredOutputTemplate(OutputFormat.Xml, defaultSchema, null);
    }

    /// <summary>
    /// Creates a CSV output template with column headers.
    /// </summary>
    /// <param name="columns">The column headers.</param>
    public static StructuredOutputTemplate Csv(params string[] columns)
    {
        var header = string.Join(",", columns);
        var schema = header + "\n<data rows>";
        return new StructuredOutputTemplate(OutputFormat.Csv, schema, null);
    }

    /// <summary>
    /// Creates a Markdown output template.
    /// </summary>
    /// <param name="structure">Optional structure guide.</param>
    public static StructuredOutputTemplate Markdown(string? structure = null)
    {
        return new StructuredOutputTemplate(OutputFormat.Markdown, structure, null);
    }

    /// <summary>
    /// Creates a YAML output template with a schema.
    /// </summary>
    /// <param name="schema">The YAML schema.</param>
    public static StructuredOutputTemplate Yaml(string? schema = null)
    {
        return new StructuredOutputTemplate(OutputFormat.Yaml, schema, null);
    }

    /// <summary>
    /// Creates a builder for constructing structured output templates.
    /// </summary>
    public static StructuredOutputBuilder Builder() => new();
}

/// <summary>
/// Builder for constructing structured output templates fluently.
/// </summary>
public class StructuredOutputBuilder
{
    private StructuredOutputTemplate.OutputFormat _format = StructuredOutputTemplate.OutputFormat.Json;
    private string? _schema;
    private string? _example;
    private string? _task;
    private readonly List<string> _fields = new();

    /// <summary>
    /// Sets the output format.
    /// </summary>
    public StructuredOutputBuilder WithFormat(StructuredOutputTemplate.OutputFormat format)
    {
        _format = format;
        return this;
    }

    /// <summary>
    /// Sets the schema definition.
    /// </summary>
    public StructuredOutputBuilder WithSchema(string schema)
    {
        _schema = schema;
        return this;
    }

    /// <summary>
    /// Sets an example of expected output.
    /// </summary>
    public StructuredOutputBuilder WithExample(string example)
    {
        _example = example;
        return this;
    }

    /// <summary>
    /// Sets the task description.
    /// </summary>
    public StructuredOutputBuilder WithTask(string task)
    {
        _task = task;
        return this;
    }

    /// <summary>
    /// Adds a field to the schema (for JSON/YAML formats).
    /// </summary>
    public StructuredOutputBuilder AddField(string name, string type)
    {
        _fields.Add($"\"{name}\": \"{type}\"");
        return this;
    }

    /// <summary>
    /// Builds the structured output template.
    /// </summary>
    public StructuredOutputTemplate Build()
    {
        // Auto-generate schema from fields if not provided
        if (string.IsNullOrWhiteSpace(_schema) && _fields.Count > 0)
        {
            if (_format == StructuredOutputTemplate.OutputFormat.Json)
            {
                _schema = "{\n  " + string.Join(",\n  ", _fields) + "\n}";
            }
            else if (_format == StructuredOutputTemplate.OutputFormat.Yaml)
            {
                _schema = string.Join("\n", _fields.Select(f => f.Replace("\"", "").Replace(": ", ": ")));
            }
        }

        var template = new StructuredOutputTemplate(_format, _schema, _example);
        if (_task is not null && !string.IsNullOrWhiteSpace(_task))
        {
            string task = _task;
            return template.WithTask(task);
        }
        return template;
    }
}
