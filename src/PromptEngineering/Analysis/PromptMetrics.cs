namespace AiDotNet.PromptEngineering.Analysis;

/// <summary>
/// Contains metrics and analysis results for a prompt.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates all the measurements and analysis data produced when
/// analyzing a prompt. It includes token counts, cost estimates, complexity scores,
/// and detected patterns that help developers understand and optimize their prompts.
/// </para>
/// <para><b>For Beginners:</b> This is a report card for your prompt.
///
/// When you analyze a prompt, you get back this object with all the measurements:
/// - Token count: How many "words" the AI sees (affects cost)
/// - Estimated cost: How much this prompt will cost in API fees
/// - Complexity score: How complicated the prompt is (0-1)
/// - Variable count: How many {placeholders} are in the prompt
/// - Detected patterns: What type of prompt this is (question, instruction, etc.)
///
/// Example usage:
/// <code>
/// var metrics = analyzer.Analyze("Translate {text} from English to Spanish");
///
/// Console.WriteLine($"Tokens: {metrics.TokenCount}");        // e.g., 8
/// Console.WriteLine($"Cost: ${metrics.EstimatedCost}");      // e.g., $0.0001
/// Console.WriteLine($"Variables: {metrics.VariableCount}");  // e.g., 1
/// Console.WriteLine($"Patterns: {string.Join(", ", metrics.DetectedPatterns)}");
/// // e.g., "translation, instruction"
/// </code>
/// </para>
/// </remarks>
public class PromptMetrics
{
    /// <summary>
    /// Gets or sets the total token count of the prompt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of tokens in the prompt as counted by the relevant tokenizer.
    /// Token count directly affects API costs and context window limits.
    /// </para>
    /// <para><b>For Beginners:</b> Tokens are like "word pieces" that AI models understand.
    ///
    /// Examples:
    /// - "Hello" = 1 token
    /// - "Hello, world!" = 3 tokens
    /// - "antidisestablishmentarianism" = 4+ tokens (long words split up)
    ///
    /// Why it matters:
    /// - API pricing is per-token
    /// - Models have maximum token limits (e.g., 4K, 8K, 128K)
    /// - More tokens = more cost and slower processing
    /// </para>
    /// </remarks>
    public int TokenCount { get; set; }

    /// <summary>
    /// Gets or sets the estimated API cost for this prompt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The estimated cost in USD for processing this prompt with the target model.
    /// Based on current API pricing and the token count.
    /// </para>
    /// <para><b>For Beginners:</b> How much money this prompt will cost.
    ///
    /// Example rates (as of 2024):
    /// - GPT-4: ~$0.03 per 1K tokens input
    /// - GPT-3.5: ~$0.001 per 1K tokens input
    /// - Claude: ~$0.008 per 1K tokens input
    ///
    /// A 500-token prompt on GPT-4 ≈ $0.015
    /// This helps you budget and optimize costs.
    /// </para>
    /// </remarks>
    public decimal EstimatedCost { get; set; }

    /// <summary>
    /// Gets or sets the complexity score of the prompt (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A normalized score indicating how complex the prompt is, considering factors
    /// like sentence structure, vocabulary diversity, nesting depth, and instruction count.
    /// </para>
    /// <para><b>For Beginners:</b> How complicated your prompt is (0 = simple, 1 = complex).
    ///
    /// Low complexity (0.0-0.3):
    /// - "What is 2+2?" (simple question)
    /// - "Say hello" (simple instruction)
    ///
    /// Medium complexity (0.3-0.7):
    /// - "Summarize this article focusing on key points"
    /// - "Translate and then explain the translation"
    ///
    /// High complexity (0.7-1.0):
    /// - "Analyze this code, identify bugs, suggest fixes, and explain your reasoning"
    /// - Multi-step instructions with conditions
    ///
    /// Complex prompts may need more capable models or clearer structure.
    /// </para>
    /// </remarks>
    public double ComplexityScore { get; set; }

    /// <summary>
    /// Gets or sets the number of template variables in the prompt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The count of variable placeholders (e.g., {variable}) found in the prompt.
    /// Useful for validating that all expected variables are present.
    /// </para>
    /// <para><b>For Beginners:</b> How many {blanks} need to be filled in.
    ///
    /// Example:
    /// "Translate {text} from {source_language} to {target_language}"
    /// VariableCount = 3
    ///
    /// This helps you:
    /// - Verify your template has the right number of variables
    /// - Catch typos in variable names
    /// - Document template requirements
    /// </para>
    /// </remarks>
    public int VariableCount { get; set; }

    /// <summary>
    /// Gets or sets the count of examples included in the prompt (for few-shot prompts).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of few-shot examples detected in the prompt. Higher example counts
    /// generally improve quality but increase token usage.
    /// </para>
    /// <para><b>For Beginners:</b> How many examples are included in your prompt.
    ///
    /// Few-shot prompts include examples to teach the AI what you want:
    /// "Translate English to Spanish:
    ///  - Hello -> Hola
    ///  - Goodbye -> Adios
    ///  Now translate: Good morning"
    ///
    /// ExampleCount = 2 (the Hello and Goodbye examples)
    ///
    /// More examples = better quality but more tokens (cost).
    /// </para>
    /// </remarks>
    public int ExampleCount { get; set; }

    /// <summary>
    /// Gets or sets the detected prompt patterns or types.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A list of patterns or prompt types detected in the prompt, such as
    /// "instruction", "question", "translation", "summarization", "chain-of-thought", etc.
    /// </para>
    /// <para><b>For Beginners:</b> What kind of prompt this is.
    ///
    /// Examples of detected patterns:
    /// - "instruction": Tells the AI to do something
    /// - "question": Asks the AI something
    /// - "translation": Asks for language translation
    /// - "summarization": Asks for a summary
    /// - "chain-of-thought": Asks AI to think step-by-step
    /// - "few-shot": Contains examples
    /// - "system-prompt": Sets AI behavior/role
    ///
    /// Knowing the pattern helps:
    /// - Choose the right model
    /// - Optimize the prompt structure
    /// - Apply appropriate preprocessing
    /// </para>
    /// </remarks>
    public IReadOnlyList<string> DetectedPatterns { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the character count of the prompt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The raw character count of the prompt string. Useful as a quick metric
    /// and for character-limited contexts.
    /// </para>
    /// <para><b>For Beginners:</b> How many letters/characters in the prompt.
    /// Similar to what you see when you check length in a text editor.
    /// </para>
    /// </remarks>
    public int CharacterCount { get; set; }

    /// <summary>
    /// Gets or sets the word count of the prompt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The approximate word count of the prompt. Note that token count is more
    /// relevant for API costs, but word count is useful for human understanding.
    /// </para>
    /// <para><b>For Beginners:</b> How many words in the prompt.
    /// A rough estimate - tokens are what actually matters for AI.
    /// Typically, 1 word ≈ 1.3 tokens on average.
    /// </para>
    /// </remarks>
    public int WordCount { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when this analysis was performed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The UTC timestamp of when the analysis was performed. Useful for caching
    /// and tracking when metrics might be stale.
    /// </para>
    /// <para><b>For Beginners:</b> When this analysis was done.
    /// Useful if you cache metrics and need to know if they're outdated.
    /// </para>
    /// </remarks>
    public DateTime AnalyzedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the name of the model used for token counting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different models use different tokenizers, so token counts can vary.
    /// This property records which model's tokenizer was used for this analysis.
    /// </para>
    /// <para><b>For Beginners:</b> Which AI model's counting method was used.
    /// GPT-4 and Claude count tokens differently, so this tells you which was used.
    /// </para>
    /// </remarks>
    public string ModelName { get; set; } = string.Empty;
}

/// <summary>
/// Represents an issue or warning detected during prompt validation.
/// </summary>
/// <remarks>
/// <para>
/// When validating a prompt, various issues may be detected. This class
/// represents a single issue with its severity, message, and location.
/// </para>
/// <para><b>For Beginners:</b> A problem found in your prompt.
///
/// Example issues:
/// - Warning: "Prompt length (5000 tokens) approaches limit (8192)"
/// - Error: "Unclosed variable placeholder at position 45"
/// - Info: "Consider adding examples for better results"
/// </para>
/// </remarks>
public class PromptIssue
{
    /// <summary>
    /// Gets or sets the severity level of the issue.
    /// </summary>
    public IssueSeverity Severity { get; set; }

    /// <summary>
    /// Gets or sets the human-readable message describing the issue.
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the issue code for programmatic handling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A short code like "PE001" that uniquely identifies this type of issue.
    /// Useful for filtering, ignoring specific issues, or looking up documentation.
    /// </para>
    /// </remarks>
    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the character position where the issue was detected (if applicable).
    /// </summary>
    public int? Position { get; set; }

    /// <summary>
    /// Gets or sets the length of the problematic text (if applicable).
    /// </summary>
    public int? Length { get; set; }
}

/// <summary>
/// Severity levels for prompt validation issues.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> How serious a problem is:
/// - Info: Just a suggestion, everything works fine
/// - Warning: Might cause problems, worth fixing
/// - Error: Will likely cause failures, must fix
/// </para>
/// </remarks>
public enum IssueSeverity
{
    /// <summary>
    /// Informational message, not a problem.
    /// </summary>
    Info,

    /// <summary>
    /// Warning that may indicate a problem.
    /// </summary>
    Warning,

    /// <summary>
    /// Error that will likely cause issues.
    /// </summary>
    Error
}

/// <summary>
/// Options for controlling prompt validation behavior.
/// </summary>
/// <remarks>
/// <para>
/// Configures what checks are performed and how strict validation should be.
/// </para>
/// <para><b>For Beginners:</b> Settings for how strict the validation should be.
///
/// Example:
/// <code>
/// var strictOptions = new ValidationOptions
/// {
///     MaxTokens = 4000,
///     CheckForInjection = true,
///     MinSeverityToReport = IssueSeverity.Info
/// };
/// </code>
/// </para>
/// </remarks>
public class ValidationOptions
{
    /// <summary>
    /// Gets or sets the maximum allowed token count.
    /// </summary>
    public int MaxTokens { get; set; } = 8192;

    /// <summary>
    /// Gets or sets whether to check for potential prompt injection.
    /// </summary>
    public bool CheckForInjection { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to validate template variables.
    /// </summary>
    public bool ValidateVariables { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum severity level to report.
    /// </summary>
    public IssueSeverity MinSeverityToReport { get; set; } = IssueSeverity.Info;

    /// <summary>
    /// Gets a strict validation configuration.
    /// </summary>
    public static ValidationOptions Strict => new()
    {
        MaxTokens = 4000,
        CheckForInjection = true,
        ValidateVariables = true,
        MinSeverityToReport = IssueSeverity.Info
    };

    /// <summary>
    /// Gets a lenient validation configuration.
    /// </summary>
    public static ValidationOptions Lenient => new()
    {
        MaxTokens = 128000,
        CheckForInjection = false,
        ValidateVariables = false,
        MinSeverityToReport = IssueSeverity.Error
    };
}
