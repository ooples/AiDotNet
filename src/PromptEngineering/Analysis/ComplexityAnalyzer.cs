using System.Text.RegularExpressions;

namespace AiDotNet.PromptEngineering.Analysis;

/// <summary>
/// Analyzer that focuses on measuring prompt complexity and structure.
/// </summary>
/// <remarks>
/// <para>
/// This analyzer provides detailed complexity metrics including readability scores,
/// structural analysis, and recommendations for simplification.
/// </para>
/// <para><b>For Beginners:</b> Measures how complicated your prompt is.
///
/// Example:
/// <code>
/// var analyzer = new ComplexityAnalyzer();
/// var metrics = analyzer.Analyze(complexPrompt);
///
/// Console.WriteLine($"Complexity: {metrics.ComplexityScore:P0}");
/// // 0% = very simple, 100% = very complex
///
/// // Access detailed breakdown:
/// if (metrics is ExtendedPromptMetrics extended)
/// {
///     Console.WriteLine($"Readability: {extended.ReadabilityScore:F1}");
///     Console.WriteLine($"Nesting depth: {extended.NestingDepth}");
/// }
/// </code>
///
/// High complexity might mean:
/// - Difficult for AI to understand
/// - Higher chance of misinterpretation
/// - May need simplification
/// </para>
/// </remarks>
public class ComplexityAnalyzer : PromptAnalyzerBase
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>

    /// <summary>
    /// Initializes a new instance of the ComplexityAnalyzer class.
    /// </summary>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public ComplexityAnalyzer(Func<string, int>? tokenCounter = null)
        : base("ComplexityAnalyzer", "general", 0.0m, tokenCounter)
    {
    }

    /// <summary>
    /// Adds detailed complexity metrics to the analysis.
    /// </summary>
    protected override void EnhanceMetrics(string prompt, PromptMetrics metrics)
    {
        // Calculate additional complexity metrics
        var readability = CalculateReadability(prompt);
        var nestingDepth = CalculateNestingDepth(prompt);
        var instructionCount = CountInstructions(prompt);
        var sentenceCount = CountSentences(prompt);

        // Adjust complexity score based on detailed analysis
        double detailedComplexity = 0.0;

        // Readability factor (higher score = easier to read = lower complexity)
        if (readability < 30) detailedComplexity += 0.3; // Very difficult
        else if (readability < 50) detailedComplexity += 0.2;
        else if (readability < 70) detailedComplexity += 0.1;

        // Nesting factor
        detailedComplexity += Math.Min(0.2, nestingDepth * 0.05);

        // Instruction density factor
        if (sentenceCount > 0)
        {
            var instructionDensity = instructionCount / (double)sentenceCount;
            detailedComplexity += Math.Min(0.2, instructionDensity * 0.2);
        }

        // Update complexity score with refined calculation
        metrics.ComplexityScore = (metrics.ComplexityScore + detailedComplexity) / 2.0;
        metrics.ComplexityScore = Math.Min(1.0, metrics.ComplexityScore);
    }

    /// <summary>
    /// Adds custom validation for complexity-related issues.
    /// </summary>
    protected override IEnumerable<PromptIssue> CustomValidation(string prompt, ValidationOptions options)
    {
        var issues = new List<PromptIssue>();

        // Check nesting depth
        var nestingDepth = CalculateNestingDepth(prompt);
        if (nestingDepth > 3)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Warning,
                Code = "CX001",
                Message = $"High nesting depth ({nestingDepth}). Consider flattening the structure."
            });
        }

        // Check sentence length
        var sentences = RegexHelper.Split(prompt, @"[.!?]+", RegexOptions.None).Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
        var longSentences = sentences.Where(s => CountWords(s) > 40).ToList();
        if (longSentences.Count > 0)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Info,
                Code = "CX002",
                Message = $"Found {longSentences.Count} very long sentence(s). Consider breaking them up."
            });
        }

        // Check instruction count
        var instructionCount = CountInstructions(prompt);
        if (instructionCount > 10)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Warning,
                Code = "CX003",
                Message = $"High instruction count ({instructionCount}). Consider consolidating or prioritizing."
            });
        }

        // Check readability
        var readability = CalculateReadability(prompt);
        if (readability < 30)
        {
            issues.Add(new PromptIssue
            {
                Severity = IssueSeverity.Info,
                Code = "CX004",
                Message = $"Low readability score ({readability:F0}). Consider simplifying language."
            });
        }

        return issues;
    }

    /// <summary>
    /// Calculates Flesch Reading Ease score (0-100, higher = easier).
    /// </summary>
    private static double CalculateReadability(string text)
    {
        var sentences = CountSentences(text);
        var words = CountWords(text);
        var syllables = CountSyllables(text);

        if (sentences == 0 || words == 0)
        {
            return 100; // Empty or very short = easy
        }

        // Flesch Reading Ease formula
        var score = 206.835
            - 1.015 * (words / (double)sentences)
            - 84.6 * (syllables / (double)words);

        return Math.Max(0, Math.Min(100, score));
    }

    /// <summary>
    /// Counts sentences in the text.
    /// </summary>
    private static int CountSentences(string text)
    {
        var sentences = RegexHelper.Split(text, @"[.!?]+", RegexOptions.None)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();
        return Math.Max(1, sentences.Count);
    }

    /// <summary>
    /// Estimates syllable count in text.
    /// </summary>
    private static int CountSyllables(string text)
    {
        var words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        return words.Sum(CountWordSyllables);
    }

    /// <summary>
    /// Estimates syllable count for a single word.
    /// </summary>
    private static int CountWordSyllables(string word)
    {
        word = word.ToLowerInvariant().Trim('\'', '"', ',', '.', '!', '?', ';', ':');

        if (string.IsNullOrWhiteSpace(word) || word.Length <= 2)
        {
            return 1;
        }

        // Count vowel groups
        var vowelGroups = RegexHelper.Matches(word, @"[aeiouy]+", RegexOptions.None).Count;

        // Adjust for silent e at end
        if (word.EndsWith("e") && vowelGroups > 1)
        {
            vowelGroups--;
        }

        // Adjust for common patterns
        if (word.EndsWith("le") && word.Length > 2 && !IsVowel(word[word.Length - 3]))
        {
            vowelGroups++;
        }

        return Math.Max(1, vowelGroups);
    }

    /// <summary>
    /// Checks if a character is a vowel.
    /// </summary>
    private static bool IsVowel(char c)
    {
        return "aeiouy".Contains(char.ToLowerInvariant(c));
    }

    /// <summary>
    /// Calculates maximum nesting depth in the prompt.
    /// </summary>
    private static int CalculateNestingDepth(string text)
    {
        int maxDepth = 0;
        int currentDepth = 0;

        foreach (var c in text)
        {
            if (c == '(' || c == '[' || c == '{' || c == '<')
            {
                currentDepth++;
                maxDepth = Math.Max(maxDepth, currentDepth);
            }
            else if (c == ')' || c == ']' || c == '}' || c == '>')
            {
                currentDepth = Math.Max(0, currentDepth - 1);
            }
        }

        return maxDepth;
    }

    /// <summary>
    /// Counts instruction-related keywords.
    /// </summary>
    private static int CountInstructions(string text)
    {
        return RegexHelper.Matches(text,
            @"\b(must|should|need to|required|ensure|make sure|do not|don't|never|always|important|note that|remember|keep in mind)\b",
            RegexOptions.IgnoreCase).Count;
    }

}



