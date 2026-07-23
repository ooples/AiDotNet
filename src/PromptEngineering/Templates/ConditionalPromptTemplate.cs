using System.Text.RegularExpressions;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Template that supports conditional sections based on variable presence or values.
/// </summary>
/// <remarks>
/// <para>
/// This template allows including or excluding sections based on conditions,
/// making prompts more dynamic and context-aware.
/// </para>
/// <para><b>For Beginners:</b> Include parts of the prompt only when conditions are met.
///
/// Example:
/// <code>
/// var template = new ConditionalPromptTemplate(@"
///     Analyze this text: {text}
///     {{#if context}}Additional context: {context}{{/if}}
///     {{#if style}}Writing style: {style}{{/if}}
/// ");
///
/// // With context
/// template.Format(new Dictionary&lt;string, string&gt;
/// {
///     { "text", "Hello world" },
///     { "context", "Programming tutorial" }
/// });
/// // Output: "Analyze this text: Hello world
/// //          Additional context: Programming tutorial"
///
/// // Without context
/// template.Format(new Dictionary&lt;string, string&gt;
/// {
///     { "text", "Hello world" }
/// });
/// // Output: "Analyze this text: Hello world"
/// </code>
/// </para>
/// </remarks>
public class ConditionalPromptTemplate : PromptTemplateBase
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private static readonly Regex ConditionalPattern = new(
        @"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}",
        RegexOptions.Compiled | RegexOptions.Singleline, RegexTimeout);

    private static readonly Regex UnlessPattern = new(
        @"\{\{#unless\s+(\w+)\}\}(.*?)\{\{/unless\}\}",
        RegexOptions.Compiled | RegexOptions.Singleline, RegexTimeout);

    private static readonly Regex EqualsPattern = new(
        @"\{\{#equals\s+(\w+)\s+""([^""]*)""\}\}(.*?)\{\{/equals\}\}",
        RegexOptions.Compiled | RegexOptions.Singleline, RegexTimeout);

    /// <summary>
    /// Initializes a new instance of the ConditionalPromptTemplate class.
    /// </summary>
    /// <param name="template">The template with conditional sections.</param>
    public ConditionalPromptTemplate(string template)
        : base(template)
    {
        // Extract additional variables from conditions
        var allVariables = new HashSet<string>(InputVariables);

        // Extract variables from #if conditions
        foreach (Match match in ConditionalPattern.Matches(template))
        {
            allVariables.Add(match.Groups[1].Value);
        }

        // Extract variables from #unless conditions
        foreach (Match match in UnlessPattern.Matches(template))
        {
            allVariables.Add(match.Groups[1].Value);
        }

        // Extract variables from #equals conditions
        foreach (Match match in EqualsPattern.Matches(template))
        {
            allVariables.Add(match.Groups[1].Value);
        }

        InputVariables = allVariables.ToList().AsReadOnly();
    }

    /// <summary>
    /// Formats the template, evaluating all conditional sections.
    /// </summary>
    protected override string FormatCore(Dictionary<string, string> variables)
    {
        var result = Template;

        // Process #equals conditions first (most specific)
        result = EqualsPattern.Replace(result, match =>
        {
            var varName = match.Groups[1].Value;
            var expectedValue = match.Groups[2].Value;
            var content = match.Groups[3].Value;

            if (variables.TryGetValue(varName, out var actualValue) &&
                string.Equals(actualValue, expectedValue, StringComparison.OrdinalIgnoreCase))
            {
                return content;
            }
            return string.Empty;
        });

        // Process #if conditions
        result = ConditionalPattern.Replace(result, match =>
        {
            var varName = match.Groups[1].Value;
            var content = match.Groups[2].Value;

            if (variables.TryGetValue(varName, out var value) &&
                !string.IsNullOrWhiteSpace(value))
            {
                return content;
            }
            return string.Empty;
        });

        // Process #unless conditions (inverse of #if)
        result = UnlessPattern.Replace(result, match =>
        {
            var varName = match.Groups[1].Value;
            var content = match.Groups[2].Value;

            if (!variables.TryGetValue(varName, out var value) ||
                string.IsNullOrWhiteSpace(value))
            {
                return content;
            }
            return string.Empty;
        });

        // Now substitute regular variables
        foreach (var kvp in variables)
        {
            var placeholder = $"{{{kvp.Key}}}";
            result = result.Replace(placeholder, kvp.Value ?? string.Empty);
        }

        // Clean up extra whitespace from removed sections
        result = Regex.Replace(result, @"\n\s*\n\s*\n", "\n\n", RegexOptions.None, RegexTimeout);

        return result.Trim();
    }

    /// <summary>
    /// Validates variables - only requires non-conditional variables.
    /// </summary>
    public override bool Validate(Dictionary<string, string> variables)
    {
        if (variables == null) return false;

        // Find required variables (those not inside conditional blocks)
        var required = ExtractRequiredVariables();
        return required.All(v => variables.ContainsKey(v) && variables[v] != null);
    }

    private HashSet<string> ExtractRequiredVariables()
    {
        // Get all variables from unconditional parts
        var tempTemplate = Template;

        // Remove all conditional sections
        tempTemplate = ConditionalPattern.Replace(tempTemplate, "");
        tempTemplate = UnlessPattern.Replace(tempTemplate, "");
        tempTemplate = EqualsPattern.Replace(tempTemplate, "");

        // Extract remaining variables
        var variablePattern = new Regex(@"\{(\w+)\}", RegexOptions.None, RegexTimeout);
        var matches = variablePattern.Matches(tempTemplate);

        return new HashSet<string>(
            matches.Cast<Match>()
                .Where(m => m.Groups.Count > 1)
                .Select(m => m.Groups[1].Value));
    }
}
