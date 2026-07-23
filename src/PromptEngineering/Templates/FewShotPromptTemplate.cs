using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Prompt template that includes few-shot examples to guide model behavior.
/// </summary>
/// <typeparam name="T">The numeric type used by the example selector for scoring and similarity.</typeparam>
/// <remarks>
/// <para>
/// This template combines a base template with examples selected from a few-shot example selector.
/// The examples are formatted and included in the prompt to demonstrate the desired input-output pattern.
/// </para>
/// <para><b>For Beginners:</b> A template that shows examples to teach the model what you want.
///
/// Example:
/// ```csharp
/// // Create base template
/// var baseTemplate = "Classify the sentiment of the following text.\n\n{examples}\n\nText: {text}\nSentiment:";
///
/// // Create few-shot template with example selector
/// var template = new FewShotPromptTemplate&lt;double&gt;(
///     baseTemplate,
///     exampleSelector,
///     exampleCount: 3,
///     exampleFormat: "Text: {input}\nSentiment: {output}"
/// );
///
/// var prompt = template.Format(new Dictionary<string, string>
/// {
///     ["text"] = "This movie was amazing!"
/// });
///
/// // Result includes 3 examples + the query:
/// // "Classify the sentiment of the following text.
/// //
/// // Text: I loved this product!
/// // Sentiment: Positive
/// //
/// // Text: Terrible experience.
/// // Sentiment: Negative
/// //
/// // Text: It's okay, nothing special.
/// // Sentiment: Neutral
/// //
/// // Text: This movie was amazing!
/// // Sentiment:"
/// ```
/// </para>
/// </remarks>
public class FewShotPromptTemplate<T> : PromptTemplateBase
{
    private readonly IFewShotExampleSelector<T> _exampleSelector;
    private readonly int _exampleCount;
    private readonly string _exampleFormat;
    private readonly string _exampleSeparator;

    /// <summary>
    /// Initializes a new instance of the FewShotPromptTemplate class.
    /// </summary>
    /// <param name="template">The base template string with {examples} placeholder.</param>
    /// <param name="exampleSelector">The selector for choosing examples.</param>
    /// <param name="exampleCount">Number of examples to include.</param>
    /// <param name="exampleFormat">Format string for each example (with {input} and {output} placeholders).</param>
    /// <param name="exampleSeparator">Separator between examples (default: double newline).</param>
    public FewShotPromptTemplate(
        string template,
        IFewShotExampleSelector<T> exampleSelector,
        int exampleCount = 3,
        string? exampleFormat = null,
        string? exampleSeparator = null)
        : base(template)
    {
        if (exampleSelector == null)
        {
            throw new ArgumentNullException(nameof(exampleSelector), "Example selector cannot be null.");
        }

        if (exampleCount <= 0)
        {
            throw new ArgumentException("Example count must be positive.", nameof(exampleCount));
        }

        _exampleSelector = exampleSelector;
        _exampleCount = exampleCount;
        _exampleFormat = exampleFormat ?? "Input: {input}\nOutput: {output}";
        _exampleSeparator = exampleSeparator ?? "\n\n";
    }

    /// <summary>
    /// Validates that the provided variables match the template's requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The 'examples' variable is excluded from validation as it is auto-generated
    /// from the example selector during formatting.
    /// </para>
    /// </remarks>
    public override bool Validate(Dictionary<string, string> variables)
    {
        if (variables == null)
        {
            return false;
        }

        // Check all required variables except 'examples' which is auto-generated
        return InputVariables
            .Where(v => v != "examples")
            .All(variable => variables.TryGetValue(variable, out var value) && value != null);
    }

    /// <summary>
    /// Formats the template with examples and variables.
    /// </summary>
    protected override string FormatCore(Dictionary<string, string> variables)
    {
        // Build the query from variables (excluding 'examples' which is auto-generated)
        var queryParts = variables
            .Where(kvp => kvp.Key != "examples")
            .Select(kvp => kvp.Value);
        var query = string.Join(" ", queryParts);

        // Select examples based on query
        var examples = _exampleSelector.SelectExamples(query, _exampleCount);

        // Format examples
        var formattedExamples = examples.Select(ex =>
        {
            return _exampleFormat
                .Replace("{input}", ex.Input)
                .Replace("{output}", ex.Output);
        });

        var examplesText = string.Join(_exampleSeparator, formattedExamples);

        // Add examples to variables
        var variablesWithExamples = new Dictionary<string, string>(variables)
        {
            ["examples"] = examplesText
        };

        // Use base formatting
        return base.FormatCore(variablesWithExamples);
    }
}
