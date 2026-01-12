using System.Text.RegularExpressions;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Base class for prompt template implementations providing common functionality and validation.
/// </summary>
/// <remarks>
/// <para>
/// This base class handles template parsing, variable extraction, validation, and formatting.
/// Derived classes can override formatting behavior for specialized template types.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all prompt templates.
///
/// It handles the common tasks:
/// - Parsing templates to find variables
/// - Validating that all required variables are provided
/// - Substituting variables into the template
/// - Error checking
///
/// When you create a new template type, inherit from this class and you get
/// all this functionality for free!
/// </para>
/// </remarks>
public abstract class PromptTemplateBase : IPromptTemplate
{
    private static readonly Regex VariablePattern = RegexHelper.Create(@"\{(\w+)\}", RegexOptions.Compiled);

    /// <summary>
    /// Gets the raw template string before variable substitution.
    /// </summary>
    public string Template { get; protected set; }

    /// <summary>
    /// Gets the list of variable names that this template expects.
    /// </summary>
    public IReadOnlyList<string> InputVariables { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the PromptTemplateBase class.
    /// </summary>
    /// <param name="template">The template string with variable placeholders.</param>
    /// <exception cref="ArgumentNullException">Thrown when template is null.</exception>
    /// <exception cref="ArgumentException">Thrown when template is empty or whitespace.</exception>
    protected PromptTemplateBase(string template)
    {
        if (template == null)
        {
            throw new ArgumentNullException(nameof(template), "Template cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(template))
        {
            throw new ArgumentException("Template cannot be empty or whitespace.", nameof(template));
        }

        Template = template;
        InputVariables = ExtractVariables(template);
    }

    /// <summary>
    /// Formats the template with the provided variables to create a complete prompt.
    /// </summary>
    /// <param name="variables">Dictionary of variable names and their values.</param>
    /// <returns>The formatted prompt string.</returns>
    /// <exception cref="ArgumentNullException">Thrown when variables is null.</exception>
    /// <exception cref="ArgumentException">Thrown when required variables are missing.</exception>
    public virtual string Format(Dictionary<string, string> variables)
    {
        if (variables == null)
        {
            throw new ArgumentNullException(nameof(variables), "Variables dictionary cannot be null.");
        }

        if (!Validate(variables))
        {
            var missing = InputVariables.Where(v => !variables.ContainsKey(v)).ToList();
            throw new ArgumentException(
                $"Missing required variables: {string.Join(", ", missing)}",
                nameof(variables));
        }

        return FormatCore(variables);
    }

    /// <summary>
    /// Validates that the provided variables match the template's requirements.
    /// </summary>
    /// <param name="variables">Dictionary of variable names and their values.</param>
    /// <returns>True if all required variables are present and valid; otherwise, false.</returns>
    public virtual bool Validate(Dictionary<string, string> variables)
    {
        if (variables == null)
        {
            return false;
        }

        // Check that all required variables are present and not null using LINQ
        return InputVariables.All(variable =>
            variables.TryGetValue(variable, out var value) && value != null);
    }

    /// <summary>
    /// Core formatting logic to be implemented by derived classes.
    /// </summary>
    /// <param name="variables">Dictionary of variable names and their values.</param>
    /// <returns>The formatted prompt string.</returns>
    /// <remarks>
    /// <para>
    /// This method is called after validation has passed. Derived classes can override
    /// this to implement custom formatting logic while relying on the base class for validation.
    /// </para>
    /// </remarks>
    protected virtual string FormatCore(Dictionary<string, string> variables)
    {
        var result = Template;

        // Replace each variable with its value
        foreach (var kvp in variables)
        {
            var placeholder = $"{{{kvp.Key}}}";
            result = result.Replace(placeholder, kvp.Value);
        }

        return result;
    }

    /// <summary>
    /// Extracts variable names from a template string.
    /// </summary>
    /// <param name="template">The template string to parse.</param>
    /// <returns>List of unique variable names found in the template.</returns>
    private static List<string> ExtractVariables(string template)
    {
        var matches = VariablePattern.Matches(template);

        var variables = new HashSet<string>(
            matches.Cast<Match>()
                .Where(match => match.Groups.Count > 1)
                .Select(match => match.Groups[1].Value));

        return variables.ToList();
    }
}



