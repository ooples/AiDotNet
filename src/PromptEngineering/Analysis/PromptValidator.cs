using System.Text.RegularExpressions;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Analysis;

/// <summary>
/// Specialized prompt validator with comprehensive validation rules.
/// </summary>
/// <remarks>
/// <para>
/// This validator performs detailed validation of prompts, checking for
/// common issues, security concerns, and best practice violations.
/// </para>
/// <para><b>For Beginners:</b> Checks your prompt for problems before you use it.
///
/// Example:
/// <code>
/// var validator = new PromptValidator();
///
/// // Check for issues
/// var issues = validator.Validate("Your prompt here {incomplete");
///
/// foreach (var issue in issues)
/// {
///     Console.WriteLine($"[{issue.Severity}] {issue.Code}: {issue.Message}");
/// }
/// // Output: [Error] PE010: Mismatched braces: 1 opening, 0 closing
/// </code>
///
/// What it checks:
/// - Syntax errors (missing braces, unclosed quotes)
/// - Security issues (potential injection attacks)
/// - Best practices (length, clarity)
/// - Model compatibility
/// </para>
/// </remarks>
public class PromptValidator
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private readonly ValidationOptions _defaultOptions;
    private readonly IPromptAnalyzer _analyzer;

    /// <summary>
    /// Initializes a new instance of the PromptValidator class.
    /// </summary>
    /// <param name="defaultOptions">Default validation options to use.</param>
    /// <param name="analyzer">Optional analyzer for additional checks.</param>
    public PromptValidator(
        ValidationOptions? defaultOptions = null,
        IPromptAnalyzer? analyzer = null)
    {
        _defaultOptions = defaultOptions ?? new ValidationOptions();
        _analyzer = analyzer ?? new TokenCountAnalyzer();
    }

    /// <summary>
    /// Validates a prompt and returns all detected issues.
    /// </summary>
    /// <param name="prompt">The prompt to validate.</param>
    /// <param name="options">Optional validation options override.</param>
    /// <returns>A list of detected issues.</returns>
    public IReadOnlyList<PromptIssue> Validate(string prompt, ValidationOptions? options = null)
    {
        var opts = options ?? _defaultOptions;
        var issues = new List<PromptIssue>();

        // Basic validation
        issues.AddRange(ValidateBasic(prompt));

        try
        {
            // Syntax validation
            issues.AddRange(ValidateSyntax(prompt));

            // Length validation
            issues.AddRange(ValidateLength(prompt, opts));

            // Security validation
            if (opts.CheckForInjection)
            {
                issues.AddRange(ValidateSecurity(prompt));
            }

            // Variable validation
            if (opts.ValidateVariables)
            {
                issues.AddRange(ValidateVariables(prompt));
            }

            // Best practice validation
            issues.AddRange(ValidateBestPractices(prompt));
        }
        catch (RegexMatchTimeoutException)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE099",
                Message = "Prompt validation timed out while evaluating regular expressions."
            });
        }

        // Filter by severity
        return issues
            .Where(i => i.Severity >= opts.MinSeverityToReport)
            .OrderByDescending(i => i.Severity)
            .ToList()
            .AsReadOnly();
    }

    /// <summary>
    /// Validates basic prompt requirements.
    /// </summary>
    private static IEnumerable<PromptIssue> ValidateBasic(string prompt)
    {
        var issues = new List<PromptIssue>();

        if (prompt == null)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE001",
                Message = "Prompt is null"
            });
            return issues;
        }

        if (string.IsNullOrWhiteSpace(prompt))
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE002",
                Message = "Prompt is empty or whitespace-only"
            });
        }

        return issues;
    }

    /// <summary>
    /// Validates prompt syntax.
    /// </summary>
    private static IEnumerable<PromptIssue> ValidateSyntax(string prompt)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            return Enumerable.Empty<PromptIssue>();
        }

        var issues = new List<PromptIssue>();

        // Check braces
        var openBraces = 0;
        var closeBraces = 0;
        for (int i = 0; i < prompt.Length; i++)
        {
            if (prompt[i] == '{') openBraces++;
            if (prompt[i] == '}') closeBraces++;
        }

        if (openBraces != closeBraces)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE010",
                Message = $"Mismatched braces: {openBraces} opening, {closeBraces} closing"
            });
        }

        // Check parentheses
        var openParens = prompt.Count(c => c == '(');
        var closeParens = prompt.Count(c => c == ')');
        if (openParens != closeParens)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Warning,
                Code = "PE011",
                Message = $"Mismatched parentheses: {openParens} opening, {closeParens} closing"
            });
        }

        // Check brackets
        var openBrackets = prompt.Count(c => c == '[');
        var closeBrackets = prompt.Count(c => c == ']');
        if (openBrackets != closeBrackets)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Warning,
                Code = "PE012",
                Message = $"Mismatched brackets: {openBrackets} opening, {closeBrackets} closing"
            });
        }

        // Check code blocks
        var codeBlockCount = Regex.Matches(prompt, "```", RegexOptions.None, RegexTimeout).Count;
        if (codeBlockCount % 2 != 0)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Warning,
                Code = "PE013",
                Message = "Unclosed code block (odd number of ``` markers)"
            });
        }

        return issues;
    }

    /// <summary>
    /// Validates prompt length.
    /// </summary>
    private IEnumerable<PromptIssue> ValidateLength(string prompt, ValidationOptions options)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            return Enumerable.Empty<PromptIssue>();
        }

        var issues = new List<PromptIssue>();
        var metrics = _analyzer.Analyze(prompt);

        if (metrics.TokenCount > options.MaxTokens)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Error,
                Code = "PE020",
                Message = $"Prompt exceeds token limit: {metrics.TokenCount} > {options.MaxTokens}"
            });
        }
        else if (metrics.TokenCount > options.MaxTokens * 0.9)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Warning,
                Code = "PE021",
                Message = $"Prompt approaching token limit: {metrics.TokenCount}/{options.MaxTokens} ({metrics.TokenCount * 100.0 / options.MaxTokens:F0}%)"
            });
        }
        else if (metrics.TokenCount > options.MaxTokens * 0.75)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Info,
                Code = "PE022",
                Message = $"Prompt using significant tokens: {metrics.TokenCount}/{options.MaxTokens} ({metrics.TokenCount * 100.0 / options.MaxTokens:F0}%)"
            });
        }

        return issues;
    }

    /// <summary>
    /// Validates prompt security.
    /// </summary>
    private static IEnumerable<PromptIssue> ValidateSecurity(string prompt)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            return Enumerable.Empty<PromptIssue>();
        }

        var issues = new List<PromptIssue>();

        // Injection patterns with descriptions
        var injectionPatterns = new[]
        {
            (@"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)", "Instruction override"),
            (@"forget\s+(everything|all)", "Memory wipe"),
            (@"you\s+are\s+now\s+", "Role hijacking"),
            (@"jailbreak", "Jailbreak keyword"),
            (@"DAN\s+mode", "DAN mode reference"),
            (@"disregard\s+(your\s+)?(rules|guidelines|instructions)", "Rule bypass"),
            (@"act\s+as\s+if.*no\s+(restrictions?|limits?)", "Restriction bypass"),
            (@"pretend\s+you\s+(can|are\s+able)", "Capability override"),
            (@"output\s+your\s+(system|initial)\s+prompt", "Prompt extraction"),
            (@"reveal\s+(your|the)\s+(system|hidden|secret)\s+prompt", "Prompt extraction"),
            (@"<\/?system>", "XML injection marker"),
            (@"\[INST\].*\[\/INST\]", "Instruction tag injection")
        };

        foreach (var (pattern, description) in injectionPatterns)
        {
            var matches = Regex.Matches(prompt, pattern, RegexOptions.IgnoreCase, RegexTimeout);
            foreach (Match match in matches)
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Warning,
                    Code = "PE030",
                    Message = $"Potential prompt injection: {description}",
                    Position = match.Index,
                    Length = match.Length
                });
            }
        }

        return issues;
    }

    /// <summary>
    /// Validates template variables.
    /// </summary>
    private static IEnumerable<PromptIssue> ValidateVariables(string prompt)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            return Enumerable.Empty<PromptIssue>();
        }

        var issues = new List<PromptIssue>();

        // Find all variables
        var variables = Regex.Matches(prompt, @"\{([^}]*)\}", RegexOptions.None, RegexTimeout);
        var seenVariables = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (Match match in variables)
        {
            var varName = match.Groups[1].Value;
            var position = match.Index;

            // Empty variable
            if (string.IsNullOrWhiteSpace(varName))
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Error,
                    Code = "PE040",
                    Message = "Empty variable name",
                    Position = position
                });
                continue;
            }

            // Variable with spaces (might be intentional, so warning)
            if (varName.Contains(" ") && !varName.Contains(":"))
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Warning,
                    Code = "PE041",
                    Message = $"Variable '{{{varName}}}' contains spaces - possible typo",
                    Position = position
                });
            }

            // Duplicate variable (info only)
            if (!seenVariables.Add(varName))
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Info,
                    Code = "PE042",
                    Message = $"Variable '{{{varName}}}' appears multiple times",
                    Position = position
                });
            }

            // Variable name with special characters (unusual)
            if (Regex.IsMatch(varName, @"[^a-zA-Z0-9_:\s-]", RegexOptions.None, RegexTimeout))
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Info,
                    Code = "PE043",
                    Message = $"Variable '{{{varName}}}' contains special characters",
                    Position = position
                });
            }
        }

        return issues;
    }

    /// <summary>
    /// Validates best practices.
    /// </summary>
    private static IEnumerable<PromptIssue> ValidateBestPractices(string prompt)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            return Enumerable.Empty<PromptIssue>();
        }

        var issues = new List<PromptIssue>();

        // Check for excessive repetition
        var words = prompt.ToLowerInvariant().Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        var wordCounts = words.GroupBy(w => w).ToDictionary(g => g.Key, g => g.Count());
        var totalWords = words.Length;

        foreach (var kvp in wordCounts)
        {
            if (kvp.Value > 5 && kvp.Value > totalWords * 0.1 && kvp.Key.Length > 3)
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Info,
                    Code = "PE050",
                    Message = $"Word '{kvp.Key}' repeated {kvp.Value} times - consider varying language"
                });
            }
        }

        // Check for very long sentences
        var sentences = Regex.Split(prompt, @"[.!?]+", RegexOptions.None, RegexTimeout).Where(s => s.Trim().Length > 0).ToList();
        foreach (var sentence in sentences)
        {
            var sentenceWords = sentence.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;
            if (sentenceWords > 50)
            {
                issues.Add(new PromptIssue
                {
                    Severity = IssueSeverity.Info,
                    Code = "PE051",
                    Message = $"Very long sentence ({sentenceWords} words) - consider breaking up for clarity"
                });
            }
        }

        // Check for missing punctuation at end
        var trimmedPrompt = prompt.TrimEnd();
        if (trimmedPrompt.Length > 20 && !Regex.IsMatch(trimmedPrompt, @"[.!?:;,}\])""']$", RegexOptions.None, RegexTimeout))
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Info,
                Code = "PE052",
                Message = "Prompt doesn't end with punctuation"
            });
        }

        return issues;
    }

    /// <summary>
    /// Gets a quick summary of validation results.
    /// </summary>
    public ValidationSummary GetSummary(string prompt, ValidationOptions? options = null)
    {
        var issues = Validate(prompt, options);
        return new ValidationSummary
        {
            IsValid = !issues.Any(i => i.Severity == IssueSeverity.Error),
            ErrorCount = issues.Count(i => i.Severity == IssueSeverity.Error),
            WarningCount = issues.Count(i => i.Severity == IssueSeverity.Warning),
            InfoCount = issues.Count(i => i.Severity == IssueSeverity.Info),
            Issues = issues
        };
    }
}

/// <summary>
/// Summary of validation results.
/// </summary>
public class ValidationSummary
{
    /// <summary>
    /// Gets or sets whether the prompt passed validation (no errors).
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets the count of errors found.
    /// </summary>
    public int ErrorCount { get; set; }

    /// <summary>
    /// Gets or sets the count of warnings found.
    /// </summary>
    public int WarningCount { get; set; }

    /// <summary>
    /// Gets or sets the count of informational issues found.
    /// </summary>
    public int InfoCount { get; set; }

    /// <summary>
    /// Gets or sets the total count of all issues.
    /// </summary>
    public int TotalCount => ErrorCount + WarningCount + InfoCount;

    /// <summary>
    /// Gets or sets the list of all issues.
    /// </summary>
    public IReadOnlyList<PromptIssue> Issues { get; set; } = new List<PromptIssue>();
}
