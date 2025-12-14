using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Template that combines multiple prompt templates in sequence.
/// </summary>
/// <remarks>
/// <para>
/// This template allows chaining multiple templates together, making it easier
/// to build complex prompts from reusable components.
/// </para>
/// <para><b>For Beginners:</b> Combines multiple templates into one.
///
/// Example:
/// <code>
/// var header = new SimplePromptTemplate("You are a {role}.");
/// var task = new SimplePromptTemplate("Your task: {task}");
/// var format = new SimplePromptTemplate("Output format: {format}");
///
/// var combined = new CompositePromptTemplate(header, task, format);
///
/// var result = combined.Format(new Dictionary&lt;string, string&gt;
/// {
///     { "role", "expert programmer" },
///     { "task", "Review this code" },
///     { "format", "List of issues" }
/// });
///
/// // Result: "You are a expert programmer.
/// //          Your task: Review this code
/// //          Output format: List of issues"
/// </code>
/// </para>
/// </remarks>
public class CompositePromptTemplate : IPromptTemplate
{
    private readonly List<IPromptTemplate> _templates;
    private readonly string _separator;

    /// <summary>
    /// Gets the raw template string (combined from all components).
    /// </summary>
    public string Template { get; }

    /// <summary>
    /// Gets the list of all variable names from all component templates.
    /// </summary>
    public IReadOnlyList<string> InputVariables { get; }

    /// <summary>
    /// Initializes a new instance of the CompositePromptTemplate class.
    /// </summary>
    /// <param name="templates">The templates to combine.</param>
    public CompositePromptTemplate(params IPromptTemplate[] templates)
        : this("\n\n", templates)
    {
    }

    /// <summary>
    /// Initializes a new instance with a custom separator.
    /// </summary>
    /// <param name="separator">String to place between templates.</param>
    /// <param name="templates">The templates to combine.</param>
    public CompositePromptTemplate(string separator, params IPromptTemplate[] templates)
    {
        _separator = separator ?? "\n\n";
        _templates = templates?.ToList() ?? new List<IPromptTemplate>();

        // Combine templates
        Template = string.Join(_separator, _templates.Select(t => t.Template));

        // Collect all unique input variables
        var variables = new HashSet<string>();
        foreach (var template in _templates)
        {
            foreach (var variable in template.InputVariables)
            {
                variables.Add(variable);
            }
        }
        InputVariables = variables.ToList().AsReadOnly();
    }

    /// <summary>
    /// Adds a template to the composite.
    /// </summary>
    /// <param name="template">The template to add.</param>
    /// <returns>This instance for chaining.</returns>
    public CompositePromptTemplate Add(IPromptTemplate template)
    {
        if (template is not null)
        {
            var templates = _templates.Concat(new[] { template }).ToArray();
            return new CompositePromptTemplate(_separator, templates);
        }
        return this;
    }

    /// <summary>
    /// Formats all templates and combines them.
    /// </summary>
    public string Format(Dictionary<string, string> variables)
    {
        if (variables == null)
        {
            throw new ArgumentNullException(nameof(variables));
        }

        var formattedParts = new List<string>();

        foreach (var template in _templates)
        {
            // Only include templates that have all their required variables
            var templateVars = template.InputVariables;
            var hasAllVars = templateVars.All(v =>
                variables.TryGetValue(v, out var val) && val != null);

            if (hasAllVars)
            {
                formattedParts.Add(template.Format(variables));
            }
        }

        return string.Join(_separator, formattedParts);
    }

    /// <summary>
    /// Validates that all required variables are present.
    /// </summary>
    public bool Validate(Dictionary<string, string> variables)
    {
        if (variables == null) return false;

        // At least one template should be formattable
        return _templates.Any(t => t.Validate(variables));
    }

    /// <summary>
    /// Creates a builder for constructing composite templates.
    /// </summary>
    public static CompositeTemplateBuilder Builder() => new();
}

/// <summary>
/// Builder for constructing composite templates fluently.
/// </summary>
public class CompositeTemplateBuilder
{
    private readonly List<IPromptTemplate> _templates = new();
    private string _separator = "\n\n";

    /// <summary>
    /// Sets the separator between templates.
    /// </summary>
    public CompositeTemplateBuilder WithSeparator(string separator)
    {
        _separator = separator;
        return this;
    }

    /// <summary>
    /// Adds a template.
    /// </summary>
    public CompositeTemplateBuilder Add(IPromptTemplate template)
    {
        if (template is not null)
        {
            _templates.Add(template);
        }
        return this;
    }

    /// <summary>
    /// Adds a simple template from a string.
    /// </summary>
    public CompositeTemplateBuilder Add(string template)
    {
        if (!string.IsNullOrWhiteSpace(template))
        {
            _templates.Add(new SimplePromptTemplate(template));
        }
        return this;
    }

    /// <summary>
    /// Builds the composite template.
    /// </summary>
    public CompositePromptTemplate Build()
    {
        return new CompositePromptTemplate(_separator, _templates.ToArray());
    }
}
