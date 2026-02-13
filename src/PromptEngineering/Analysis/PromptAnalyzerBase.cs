using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.PromptEngineering.Analysis;

/// <summary>
/// Provides a base implementation for prompt analyzers with common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements the IPromptAnalyzer interface and provides common functionality
/// for prompt analysis. It handles validation, token counting, and delegates to derived classes
/// for specific analysis logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all prompt analyzers build upon.
///
/// Think of it like a template for examining prompts:
/// - It handles common tasks (counting words, finding variables)
/// - Specific analyzers fill in details like token counting and pattern detection
/// - This ensures all analyzers work consistently
/// </para>
/// </remarks>
public abstract class PromptAnalyzerBase : IPromptAnalyzer
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private readonly Func<string, int>? _tokenCounter;
    private readonly decimal _costPerThousandTokens;
    private readonly string _modelName;

    /// <summary>
    /// Initializes a new instance of the PromptAnalyzerBase class.
    /// </summary>
    /// <param name="name">The name of this analyzer.</param>
    /// <param name="modelName">The model this analyzer targets.</param>
    /// <param name="costPerThousandTokens">Cost per 1000 tokens for cost estimation.</param>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    protected PromptAnalyzerBase(
        string name,
        string modelName = "gpt-4",
        decimal costPerThousandTokens = 0.03m,
        Func<string, int>? tokenCounter = null)
    {
        Guard.NotNull(name);
        Name = name;
        _modelName = modelName;
        _costPerThousandTokens = costPerThousandTokens;
        _tokenCounter = tokenCounter;
    }

    /// <summary>
    /// Gets the name of this analyzer implementation.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Analyzes a prompt and returns detailed metrics.
    /// </summary>
    /// <param name="prompt">The prompt string to analyze.</param>
    /// <returns>A PromptMetrics object containing analysis results.</returns>
    public PromptMetrics Analyze(string prompt)
    {
        if (prompt == null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }

        var tokenCount = CountTokens(prompt);
        var characterCount = prompt.Length;
        var wordCount = CountWords(prompt);
        var variableCount = CountVariables(prompt);
        var exampleCount = CountExamples(prompt);
        var detectedPatterns = DetectPatterns(prompt);
        var complexityScore = CalculateComplexity(prompt, tokenCount, variableCount, exampleCount);
        var estimatedCost = CalculateCost(tokenCount);

        var metrics = new PromptMetrics
        {
            TokenCount = tokenCount,
            CharacterCount = characterCount,
            WordCount = wordCount,
            VariableCount = variableCount,
            ExampleCount = exampleCount,
            ComplexityScore = complexityScore,
            EstimatedCost = estimatedCost,
            DetectedPatterns = detectedPatterns,
            ModelName = _modelName,
            AnalyzedAt = DateTime.UtcNow
        };

        // Allow derived classes to add additional analysis
        EnhanceMetrics(prompt, metrics);

        return metrics;
    }

    /// <summary>
    /// Analyzes a prompt asynchronously.
    /// </summary>
    public virtual Task<PromptMetrics> AnalyzeAsync(string prompt, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(Analyze(prompt));
    }

    /// <summary>
    /// Validates a prompt for potential issues.
    /// </summary>
    public IEnumerable<PromptIssue> ValidatePrompt(string prompt, ValidationOptions? options = null)
    {
        var opts = options ?? new ValidationOptions();
        var issues = new List<PromptIssue>();

        if (string.IsNullOrWhiteSpace(prompt))
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE001",
                Message = "Prompt is empty or whitespace-only"
            });
            return FilterBySeverity(issues, opts.MinSeverityToReport);
        }

        // Check token limit
        var tokenCount = CountTokens(prompt);
        if (tokenCount > opts.MaxTokens)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE002",
                Message = $"Prompt exceeds maximum token limit ({tokenCount} > {opts.MaxTokens})"
            });
        }
        else if (tokenCount > opts.MaxTokens * 0.9)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Warning,
                Code = "PE003",
                Message = $"Prompt is approaching token limit ({tokenCount}/{opts.MaxTokens}, {tokenCount * 100.0 / opts.MaxTokens:F0}%)"
            });
        }

        // Validate variables
        if (opts.ValidateVariables)
        {
            issues.AddRange(ValidateVariables(prompt));
        }

        // Check for prompt injection
        if (opts.CheckForInjection)
        {
            issues.AddRange(CheckForInjection(prompt));
        }

        // Add custom validation from derived classes
        issues.AddRange(CustomValidation(prompt, opts));

        return FilterBySeverity(issues, opts.MinSeverityToReport);
    }

    /// <summary>
    /// Counts tokens in the given text.
    /// </summary>
    protected int CountTokens(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return 0;
        }

        if (_tokenCounter != null)
        {
            return _tokenCounter(text);
        }

        // Default approximation: ~0.75 tokens per word for English
        var words = CountWords(text);
        return (int)Math.Ceiling(words / 0.75);
    }

    /// <summary>
    /// Counts words in the text.
    /// </summary>
    protected static int CountWords(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return 0;
        }

        return text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;
    }

    /// <summary>
    /// Counts template variables in the prompt.
    /// </summary>
    protected static int CountVariables(string prompt)
    {
        var matches = Regex.Matches(prompt, @"\{[^}]+\}", RegexOptions.None, RegexTimeout);
        return matches.Count;
    }

    /// <summary>
    /// Counts few-shot examples in the prompt.
    /// </summary>
    protected virtual int CountExamples(string prompt)
    {
        // Look for common example patterns
        var patterns = new[]
        {
            @"Example\s*\d*\s*:",
            @"Input:\s*.*\s*Output:",
            @"Q:\s*.*\s*A:",
            @"User:\s*.*\s*Assistant:",
            @"->\s+",
            @"=>\s+"
        };

        int count = 0;
        foreach (var pattern in patterns)
        {
            count += Regex.Matches(prompt, pattern, RegexOptions.IgnoreCase, RegexTimeout).Count;
        }

        return count;
    }

    /// <summary>
    /// Detects prompt patterns/types.
    /// </summary>
    protected virtual IReadOnlyList<string> DetectPatterns(string prompt)
    {
        var patterns = new List<string>();
        var lowerPrompt = prompt.ToLowerInvariant();

        // Question detection
        if (Regex.IsMatch(prompt, @"\?(?:\s|$)", RegexOptions.None, RegexTimeout) ||
            Regex.IsMatch(lowerPrompt, @"\b(what|who|where|when|why|how|which|can you|could you|do you)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("question");
        }

        // Instruction detection
        if (Regex.IsMatch(lowerPrompt, @"\b(write|create|generate|produce|make|build|design)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("generation");
        }

        if (Regex.IsMatch(lowerPrompt, @"\b(summarize|summarise|summary|tldr|condense)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("summarization");
        }

        if (Regex.IsMatch(lowerPrompt, @"\b(translate|translation|convert.*to|from.*to)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("translation");
        }

        if (Regex.IsMatch(lowerPrompt, @"\b(analyze|analyse|analysis|examine|evaluate|assess)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("analysis");
        }

        if (Regex.IsMatch(lowerPrompt, @"\b(extract|identify|find|list|get)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("extraction");
        }

        if (Regex.IsMatch(lowerPrompt, @"\b(step.?by.?step|first.*then|reasoning|think.*through|let'?s think)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("chain-of-thought");
        }

        if (Regex.IsMatch(lowerPrompt, @"\b(you are|act as|pretend|role|persona)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("role-playing");
        }

        if (CountExamples(prompt) > 0)
        {
            patterns.Add("few-shot");
        }

        if (CountVariables(prompt) > 0)
        {
            patterns.Add("template");
        }

        if (patterns.Count == 0)
        {
            patterns.Add("general");
        }

        return patterns.AsReadOnly();
    }

    /// <summary>
    /// Calculates complexity score (0.0 to 1.0).
    /// </summary>
    protected virtual double CalculateComplexity(string prompt, int tokenCount, int variableCount, int exampleCount)
    {
        // Base complexity on various factors
        double score = 0.0;

        // Length factor (0-0.3)
        if (tokenCount > 1000) score += 0.3;
        else if (tokenCount > 500) score += 0.2;
        else if (tokenCount > 100) score += 0.1;

        // Sentence complexity (0-0.2)
        var sentences = Regex.Split(prompt, @"[.!?]+", RegexOptions.None, RegexTimeout).Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
        var avgWordsPerSentence = sentences.Count > 0 ? CountWords(prompt) / (double)sentences.Count : 0;
        if (avgWordsPerSentence > 25) score += 0.2;
        else if (avgWordsPerSentence > 15) score += 0.1;

        // Nested structure (0-0.2)
        var nestedPatterns = Regex.Matches(prompt, @"[\[\(\{<].*?[\]\)\}>]", RegexOptions.None, RegexTimeout).Count;
        if (nestedPatterns > 5) score += 0.2;
        else if (nestedPatterns > 2) score += 0.1;

        // Instruction count (0-0.2)
        var instructionKeywords = Regex.Matches(prompt,
            @"\b(must|should|need to|required|ensure|make sure|do not|don't|never|always|important)\b",
            RegexOptions.IgnoreCase, RegexTimeout).Count;
        if (instructionKeywords > 5) score += 0.2;
        else if (instructionKeywords > 2) score += 0.1;

        // Variables and examples add complexity (0-0.1)
        if (variableCount > 3 || exampleCount > 3) score += 0.1;
        else if (variableCount > 0 || exampleCount > 0) score += 0.05;

        return Math.Min(1.0, score);
    }

    /// <summary>
    /// Calculates estimated cost based on token count.
    /// </summary>
    protected decimal CalculateCost(int tokenCount)
    {
        return (tokenCount / 1000.0m) * _costPerThousandTokens;
    }

    /// <summary>
    /// Validates template variables in the prompt.
    /// </summary>
    protected virtual IEnumerable<PromptIssue> ValidateVariables(string prompt)
    {
        var issues = new List<PromptIssue>();

        // Check for unclosed braces
        var openBraces = prompt.Count(c => c == '{');
        var closeBraces = prompt.Count(c => c == '}');

        if (openBraces != closeBraces)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE010",
                Message = $"Mismatched braces: {openBraces} opening, {closeBraces} closing"
            });
        }

        // Check for empty variable names
        var emptyVars = Regex.Matches(prompt, @"\{\s*\}", RegexOptions.None, RegexTimeout);
        if (emptyVars.Count > 0)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE011",
                Message = $"Found {emptyVars.Count} empty variable placeholder(s)"
            });
        }

        // Check for suspicious variable names (potential typos)
        var vars = Regex.Matches(prompt, @"\{([^}]+)\}", RegexOptions.None, RegexTimeout);
        foreach (Match match in vars)
        {
            var varName = match.Groups[1].Value;
            if (varName.Contains(" ") && !varName.Contains(":"))
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Warning,
                    Code = "PE012",
                    Message = $"Variable '{{{{varName}}}}' contains spaces - possible typo",
                    Position = match.Index
                });
            }
        }

        return issues;
    }

    /// <summary>
    /// Checks for potential prompt injection patterns.
    /// </summary>
    protected virtual IEnumerable<PromptIssue> CheckForInjection(string prompt)
    {
        var issues = new List<PromptIssue>();

        // Common injection patterns
        var injectionPatterns = new[]
        {
            (@"\bignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)\b", "Instruction override attempt"),
            (@"\bforget\s+(everything|all)\b", "Memory manipulation attempt"),
            (@"\byou\s+are\s+now\s+", "Role hijacking attempt"),
            (@"\bjailbreak\b", "Jailbreak keyword detected"),
            (@"\bDAN\s+mode\b", "DAN mode reference detected"),
            (@"\bdisregard\s+(your\s+)?(rules|guidelines|instructions)\b", "Rule disregard attempt"),
            (@"\bact\s+as\s+if\s+you\s+(have\s+)?no\s+(restrictions?|limits?)\b", "Restriction bypass attempt")
        };

        foreach (var (pattern, description) in injectionPatterns)
        {
            var matches = Regex.Matches(prompt, pattern, RegexOptions.IgnoreCase, RegexTimeout);
            foreach (Match match in matches)
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Warning,
                    Code = "PE020",
                    Message = $"Potential prompt injection: {description}",
                    Position = match.Index,
                    Length = match.Length
                });
            }
        }

        return issues;
    }

    /// <summary>
    /// Allows derived classes to add custom metrics.
    /// </summary>
    protected virtual void EnhanceMetrics(string prompt, PromptMetrics metrics)
    {
        // Override in derived classes to add custom analysis
    }

    /// <summary>
    /// Allows derived classes to add custom validation.
    /// </summary>
    protected virtual IEnumerable<PromptIssue> CustomValidation(string prompt, ValidationOptions options)
    {
        return Enumerable.Empty<PromptIssue>();
    }

    /// <summary>
    /// Filters issues by minimum severity.
    /// </summary>
    private static IEnumerable<PromptIssue> FilterBySeverity(IEnumerable<PromptIssue> issues, IssueSeverity minSeverity)
    {
        return issues.Where(i => i.Severity >= minSeverity);
    }
}
