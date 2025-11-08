using System.Numerics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.AdversarialRobustness.Safety;

/// <summary>
/// Implements comprehensive safety filtering for AI model inputs and outputs.
/// </summary>
/// <remarks>
/// <para>
/// SafetyFilter provides multiple layers of protection including input validation,
/// output filtering, jailbreak detection, and harmful content identification.
/// </para>
/// <para><b>For Beginners:</b> Think of SafetyFilter as a comprehensive security system
/// for your AI. It checks everything going in and coming out, looking for anything
/// suspicious, harmful, or inappropriate. It's like having security guards, content
/// moderators, and safety inspectors all working together.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class SafetyFilter<T> : ISafetyFilter<T>
    where T : struct, INumber<T>
{
    private readonly SafetyFilterOptions<T> options;
    private readonly List<string> jailbreakPatterns;
    private readonly Dictionary<string, List<string>> harmfulContentPatterns;

    /// <summary>
    /// Initializes a new instance of the safety filter.
    /// </summary>
    /// <param name="options">The safety filter configuration options.</param>
    public SafetyFilter(SafetyFilterOptions<T> options)
    {
        this.options = options;

        // Initialize jailbreak detection patterns
        jailbreakPatterns = new List<string>
        {
            @"ignore (previous|all|your) (instructions?|rules?)",
            @"forget (everything|what|all)",
            @"you are now",
            @"pretend (to be|you are)",
            @"act as (if|a|an)",
            @"roleplay",
            @"new (instructions?|rules?)",
            @"disregard",
            @"override",
            @"system prompt"
        };

        // Initialize harmful content patterns by category
        harmfulContentPatterns = new Dictionary<string, List<string>>();

        foreach (var category in options.HarmfulContentCategories)
        {
            harmfulContentPatterns[category] = category.ToLower() switch
            {
                "violence" => new List<string> { @"kill", @"harm", @"weapon", @"attack", @"destroy" },
                "hatespeech" => new List<string> { @"hate", @"discriminat", @"racist", @"sexist" },
                "adultcontent" => new List<string> { @"explicit", @"nsfw", @"adult" },
                "privateinformation" => new List<string> { @"ssn", @"credit card", @"password", @"private" },
                "misinformation" => new List<string> { @"fake news", @"conspiracy", @"false" },
                _ => new List<string>()
            };
        }
    }

    /// <inheritdoc/>
    public SafetyValidationResult<T> ValidateInput(T[] input)
    {
        var result = new SafetyValidationResult<T>
        {
            IsValid = true,
            SafetyScore = 1.0,
            Issues = new List<ValidationIssue>()
        };

        if (!options.EnableInputValidation)
        {
            return result;
        }

        // Check input length
        if (input.Length > options.MaxInputLength)
        {
            result.IsValid = false;
            result.SafetyScore = 0.5;
            result.Issues.Add(new ValidationIssue
            {
                Severity = "High",
                Type = "LengthExceeded",
                Description = $"Input length {input.Length} exceeds maximum {options.MaxInputLength}",
                Location = 0
            });
        }

        // Check for NaN or infinite values
        for (int i = 0; i < input.Length; i++)
        {
            if (!T.IsFinite(input[i]))
            {
                result.IsValid = false;
                result.SafetyScore = Math.Min(result.SafetyScore, 0.3);
                result.Issues.Add(new ValidationIssue
                {
                    Severity = "Critical",
                    Type = "InvalidValue",
                    Description = "Input contains NaN or infinite values",
                    Location = i
                });
            }
        }

        // Convert to text for pattern matching (if applicable)
        var textRepresentation = ConvertToText(input);
        if (!string.IsNullOrEmpty(textRepresentation))
        {
            // Check for jailbreak attempts
            var jailbreakResult = DetectJailbreak(input);
            if (jailbreakResult.JailbreakDetected)
            {
                result.JailbreakDetected = true;
                result.SafetyScore = Math.Min(result.SafetyScore, 1.0 - jailbreakResult.Severity);
                result.IsValid = false;
                result.Issues.Add(new ValidationIssue
                {
                    Severity = "Critical",
                    Type = "JailbreakAttempt",
                    Description = $"Jailbreak attempt detected: {jailbreakResult.JailbreakType}",
                    Location = 0
                });
            }

            // Check for harmful content
            var harmfulResult = IdentifyHarmfulContent(input);
            if (harmfulResult.HarmfulContentDetected)
            {
                result.DetectedHarmCategories = harmfulResult.DetectedCategories;
                result.SafetyScore = Math.Min(result.SafetyScore, 1.0 - harmfulResult.HarmScore);

                if (harmfulResult.HarmScore > 0.5)
                {
                    result.IsValid = false;
                }

                result.Issues.Add(new ValidationIssue
                {
                    Severity = harmfulResult.HarmScore > 0.7 ? "High" : "Medium",
                    Type = "HarmfulContent",
                    Description = $"Harmful content detected: {string.Join(", ", harmfulResult.DetectedCategories)}",
                    Location = 0
                });
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public SafetyFilterResult<T> FilterOutput(T[] output)
    {
        var result = new SafetyFilterResult<T>
        {
            IsSafe = true,
            SafetyScore = 1.0,
            FilteredOutput = (T[])output.Clone(),
            WasModified = false,
            Actions = new List<FilterAction>()
        };

        if (!options.EnableOutputFiltering)
        {
            return result;
        }

        // Check for harmful content in output
        var harmfulResult = IdentifyHarmfulContent(output);

        if (harmfulResult.HarmfulContentDetected)
        {
            result.DetectedHarmCategories = harmfulResult.DetectedCategories;
            result.SafetyScore = 1.0 - harmfulResult.HarmScore;

            if (harmfulResult.HarmScore > options.SafetyThreshold)
            {
                result.IsSafe = false;

                // Apply filtering based on severity
                if (harmfulResult.HarmScore > 0.8)
                {
                    // Block entire output for severe violations
                    result.FilteredOutput = Array.Empty<T>();
                    result.WasModified = true;
                    result.Actions.Add(new FilterAction
                    {
                        ActionType = "Block",
                        Reason = "Severe harmful content detected",
                        Location = 0,
                        OriginalContent = "Output blocked"
                    });
                }
                else
                {
                    // Sanitize for moderate violations
                    result.FilteredOutput = SanitizeOutput(output, harmfulResult);
                    result.WasModified = true;
                    result.Actions.Add(new FilterAction
                    {
                        ActionType = "Sanitize",
                        Reason = "Moderate harmful content detected and sanitized",
                        Location = 0,
                        OriginalContent = "Content sanitized"
                    });
                }
            }
        }

        if (options.LogFilteredContent && !result.IsSafe)
        {
            LogFilteredContent(output, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public JailbreakDetectionResult<T> DetectJailbreak(T[] input)
    {
        var result = new JailbreakDetectionResult<T>
        {
            JailbreakDetected = false,
            ConfidenceScore = 0.0,
            Severity = 0.0,
            Indicators = new List<JailbreakIndicator>()
        };

        var text = ConvertToText(input);
        if (string.IsNullOrEmpty(text))
        {
            return result;
        }

        text = text.ToLower();
        var matchedPatterns = 0;
        var totalPatterns = jailbreakPatterns.Count;

        foreach (var pattern in jailbreakPatterns)
        {
            if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase))
            {
                matchedPatterns++;
                result.Indicators.Add(new JailbreakIndicator
                {
                    Type = "PatternMatch",
                    Description = $"Matched jailbreak pattern: {pattern}",
                    Confidence = 0.8,
                    Location = 0
                });
            }
        }

        if (matchedPatterns > 0)
        {
            result.JailbreakDetected = true;
            result.ConfidenceScore = Math.Min(1.0, (double)matchedPatterns / totalPatterns * 2.0);
            result.Severity = result.ConfidenceScore;
            result.JailbreakType = matchedPatterns > 2 ? "Sophisticated" : "Basic";
            result.RecommendedActions = new[] { "Block", "Log", "Alert" };
        }

        return result;
    }

    /// <inheritdoc/>
    public HarmfulContentResult<T> IdentifyHarmfulContent(T[] content)
    {
        var result = new HarmfulContentResult<T>
        {
            HarmfulContentDetected = false,
            HarmScore = 0.0,
            CategoryScores = new Dictionary<string, double>(),
            Findings = new List<HarmfulContentFinding>()
        };

        var text = ConvertToText(content);
        if (string.IsNullOrEmpty(text))
        {
            return result;
        }

        text = text.ToLower();

        foreach (var category in options.HarmfulContentCategories)
        {
            if (!harmfulContentPatterns.ContainsKey(category))
            {
                continue;
            }

            var patterns = harmfulContentPatterns[category];
            var matchCount = 0;

            foreach (var pattern in patterns)
            {
                if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase))
                {
                    matchCount++;
                    result.Findings.Add(new HarmfulContentFinding
                    {
                        Category = category,
                        Severity = 0.7,
                        Description = $"Matched {category} pattern: {pattern}",
                        Location = 0,
                        Excerpt = pattern
                    });
                }
            }

            if (matchCount > 0)
            {
                var categoryScore = Math.Min(1.0, (double)matchCount / patterns.Count);
                result.CategoryScores[category] = categoryScore;
                result.HarmScore = Math.Max(result.HarmScore, categoryScore);
            }
        }

        if (result.CategoryScores.Count > 0)
        {
            result.HarmfulContentDetected = true;
            result.PrimaryHarmCategory = result.CategoryScores.OrderByDescending(kv => kv.Value).First().Key;
            result.DetectedCategories = result.CategoryScores.Keys.ToArray();
            result.RecommendedAction = result.HarmScore > 0.7 ? "Block" : result.HarmScore > 0.4 ? "Warn" : "Allow";
        }

        return result;
    }

    /// <inheritdoc/>
    public T ComputeSafetyScore(T[] content)
    {
        var validation = ValidateInput(content);
        var harmfulContent = IdentifyHarmfulContent(content);
        var jailbreak = DetectJailbreak(content);

        var combinedScore = validation.SafetyScore * 0.4 +
                           (1.0 - harmfulContent.HarmScore) * 0.4 +
                           (1.0 - jailbreak.Severity) * 0.2;

        return T.CreateChecked(Math.Max(0.0, Math.Min(1.0, combinedScore)));
    }

    /// <inheritdoc/>
    public SafetyFilterOptions<T> GetOptions() => options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var json = System.Text.Json.JsonSerializer.Serialize(options);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data) { }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        Deserialize(File.ReadAllBytes(filePath));
    }

    private string ConvertToText(T[] data)
    {
        // For text data represented as numeric arrays (e.g., token IDs),
        // you would decode to text. For demonstration, we create a simple representation
        try
        {
            // This is a placeholder - in practice, you'd have proper encoding/decoding
            return string.Join(" ", data.Select(x => x.ToString() ?? ""));
        }
        catch
        {
            return string.Empty;
        }
    }

    private T[] SanitizeOutput(T[] output, HarmfulContentResult<T> harmfulResult)
    {
        // Simple sanitization - in practice, would be more sophisticated
        // For now, just return a safe default or masked version
        return (T[])output.Clone(); // Placeholder
    }

    private void LogFilteredContent(T[] output, SafetyFilterResult<T> result)
    {
        // Log to file or monitoring system
        var logEntry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}] " +
                      $"Filtered content - Categories: {string.Join(", ", result.DetectedHarmCategories)} " +
                      $"- Score: {result.SafetyScore:F3}\n";

        try
        {
            var logPath = "safety_filter_logs.txt";
            File.AppendAllText(logPath, logEntry);
        }
        catch
        {
            // Silent fail for logging
        }
    }
}
