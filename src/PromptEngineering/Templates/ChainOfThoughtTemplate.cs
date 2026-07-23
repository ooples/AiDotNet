using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Template that structures prompts for chain-of-thought reasoning.
/// </summary>
/// <remarks>
/// <para>
/// This template encourages the model to show its reasoning process step by step,
/// which often leads to better accuracy on complex tasks.
/// </para>
/// <para><b>For Beginners:</b> Helps AI think through problems step by step.
///
/// Example:
/// <code>
/// var template = new ChainOfThoughtTemplate(
///     question: "What is 23 * 17?",
///     context: "This is a multiplication problem."
/// );
///
/// var prompt = template.Format(new Dictionary&lt;string, string&gt;());
/// // Output includes structured reasoning guidance
/// </code>
///
/// How it works:
/// - Structures the prompt to encourage step-by-step thinking
/// - Includes explicit instructions for showing reasoning
/// - Helps with math, logic, and complex analysis tasks
/// </para>
/// </remarks>
public class ChainOfThoughtTemplate : PromptTemplateBase
{
    private const string DefaultTemplate = @"Question: {question}

{context_section}Think through this step by step:

1. First, understand what is being asked.
2. Break down the problem into smaller parts.
3. Work through each part systematically.
4. Verify your reasoning at each step.
5. Combine your findings into a final answer.

Let's think step by step:";

    private const string WithExamplesTemplate = @"I'll demonstrate step-by-step reasoning with examples, then apply it to a new problem.

{examples_section}Now, let's apply the same careful reasoning:

Question: {question}

{context_section}Let's think step by step:";

    private readonly List<ChainOfThoughtExample> _examples;

    /// <summary>
    /// Initializes a new instance of the ChainOfThoughtTemplate class.
    /// </summary>
    /// <param name="question">The question to reason about.</param>
    /// <param name="context">Optional additional context.</param>
    public ChainOfThoughtTemplate(string question, string? context = null)
        : base(BuildTemplate(question, context, null))
    {
        _examples = new List<ChainOfThoughtExample>();
    }

    /// <summary>
    /// Initializes a new instance with examples for few-shot chain-of-thought.
    /// </summary>
    /// <param name="question">The question to reason about.</param>
    /// <param name="examples">Examples showing step-by-step reasoning.</param>
    /// <param name="context">Optional additional context.</param>
    public ChainOfThoughtTemplate(string question, IEnumerable<ChainOfThoughtExample> examples, string? context = null)
        : base(BuildTemplateWithExamples(question, examples, context))
    {
        _examples = examples?.ToList() ?? new List<ChainOfThoughtExample>();
    }

    /// <summary>
    /// Initializes a new instance with a custom template.
    /// </summary>
    /// <param name="template">The custom chain-of-thought template.</param>
    public ChainOfThoughtTemplate(string template)
        : base(template)
    {
        _examples = new List<ChainOfThoughtExample>();
    }

    private static string BuildTemplate(string question, string? context, List<ChainOfThoughtExample>? examples)
    {
        var contextSection = string.IsNullOrWhiteSpace(context)
            ? ""
            : $"Context: {context}\n\n";

        return DefaultTemplate
            .Replace("{question}", question ?? "{question}")
            .Replace("{context_section}", contextSection);
    }

    private static string BuildTemplateWithExamples(string question, IEnumerable<ChainOfThoughtExample>? examples, string? context)
    {
        var examplesList = examples?.ToList() ?? new List<ChainOfThoughtExample>();
        var examplesSection = "";

        if (examplesList.Count > 0)
        {
            var exampleStrings = examplesList.Select((ex, i) =>
                $"Example {i + 1}:\nQuestion: {ex.Question}\nReasoning:\n{ex.Reasoning}\nAnswer: {ex.Answer}\n");
            examplesSection = string.Join("\n", exampleStrings) + "\n";
        }

        var contextSection = string.IsNullOrWhiteSpace(context)
            ? ""
            : $"Context: {context}\n\n";

        return WithExamplesTemplate
            .Replace("{question}", question ?? "{question}")
            .Replace("{examples_section}", examplesSection)
            .Replace("{context_section}", contextSection);
    }

    /// <summary>
    /// Formats the chain-of-thought template.
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
    /// Creates a builder for constructing chain-of-thought prompts.
    /// </summary>
    public static ChainOfThoughtBuilder Builder() => new();
}

/// <summary>
/// Represents an example for few-shot chain-of-thought prompting.
/// </summary>
public class ChainOfThoughtExample
{
    /// <summary>
    /// The example question.
    /// </summary>
    public string Question { get; set; } = string.Empty;

    /// <summary>
    /// The step-by-step reasoning for the example.
    /// </summary>
    public string Reasoning { get; set; } = string.Empty;

    /// <summary>
    /// The final answer for the example.
    /// </summary>
    public string Answer { get; set; } = string.Empty;

    /// <summary>
    /// Creates a new chain-of-thought example.
    /// </summary>
    public ChainOfThoughtExample() { }

    /// <summary>
    /// Creates a new chain-of-thought example with values.
    /// </summary>
    public ChainOfThoughtExample(string question, string reasoning, string answer)
    {
        Question = question;
        Reasoning = reasoning;
        Answer = answer;
    }
}

/// <summary>
/// Builder for constructing chain-of-thought templates fluently.
/// </summary>
public class ChainOfThoughtBuilder
{
    private string _question = string.Empty;
    private string? _context;
    private readonly List<ChainOfThoughtExample> _examples = new();

    /// <summary>
    /// Sets the question to reason about.
    /// </summary>
    public ChainOfThoughtBuilder WithQuestion(string question)
    {
        _question = question;
        return this;
    }

    /// <summary>
    /// Sets additional context for the question.
    /// </summary>
    public ChainOfThoughtBuilder WithContext(string context)
    {
        _context = context;
        return this;
    }

    /// <summary>
    /// Adds an example demonstrating step-by-step reasoning.
    /// </summary>
    public ChainOfThoughtBuilder AddExample(string question, string reasoning, string answer)
    {
        _examples.Add(new ChainOfThoughtExample(question, reasoning, answer));
        return this;
    }

    /// <summary>
    /// Adds an example demonstrating step-by-step reasoning.
    /// </summary>
    public ChainOfThoughtBuilder AddExample(ChainOfThoughtExample example)
    {
        if (example is not null)
        {
            _examples.Add(example);
        }
        return this;
    }

    /// <summary>
    /// Builds the chain-of-thought template.
    /// </summary>
    public ChainOfThoughtTemplate Build()
    {
        if (_examples.Count > 0)
        {
            return new ChainOfThoughtTemplate(_question, _examples, _context);
        }
        return new ChainOfThoughtTemplate(_question, _context);
    }
}
