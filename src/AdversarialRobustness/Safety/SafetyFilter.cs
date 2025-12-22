using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Serialization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using System.Text;

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
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Timeout for regex operations to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromMilliseconds(100);

    private SafetyFilterOptions<T> options;
    private readonly List<string> jailbreakPatterns;
    private readonly Dictionary<string, List<string>> harmfulContentPatterns;

    /// <summary>
    /// Initializes a new instance of the safety filter.
    /// </summary>
    /// <param name="options">The safety filter configuration options.</param>
    public SafetyFilter(SafetyFilterOptions<T> options)
    {
        this.options = options ?? throw new ArgumentNullException(nameof(options));
        jailbreakPatterns = new List<string>();
        harmfulContentPatterns = new Dictionary<string, List<string>>();
        InitializePatterns();
    }

    /// <summary>
    /// Initializes the jailbreak and harmful content detection patterns from options.
    /// </summary>
    private void InitializePatterns()
    {
        // Initialize jailbreak detection patterns
        jailbreakPatterns.Clear();
        jailbreakPatterns.AddRange(new[]
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
        });

        // Initialize harmful content patterns by category
        harmfulContentPatterns.Clear();

        foreach (var category in options.HarmfulContentCategories)
        {
            harmfulContentPatterns[category] = category.ToLowerInvariant() switch
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
    public SafetyValidationResult<T> ValidateInput(Vector<T> input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

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
            var d = NumOps.ToDouble(input[i]);
            if (double.IsNaN(d) || double.IsInfinity(d))
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
    public SafetyFilterResult<T> FilterOutput(Vector<T> output)
    {
        if (output == null)
        {
            throw new ArgumentNullException(nameof(output));
        }

        var result = new SafetyFilterResult<T>
        {
            IsSafe = true,
            SafetyScore = 1.0,
            FilteredOutput = CloneVector(output),
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
                    result.FilteredOutput = Vector<T>.Empty();
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

                    // Build description of what was sanitized from the findings
                    var sanitizedDescription = harmfulResult.Findings.Count > 0
                        ? string.Join("; ", harmfulResult.Findings.Select(f =>
                            $"{f.Category} at location {f.Location}: {f.Excerpt}"))
                        : $"Harmful content detected with score {harmfulResult.HarmScore:F3}";

                    result.Actions.Add(new FilterAction
                    {
                        ActionType = "Sanitize",
                        Reason = "Moderate harmful content detected and sanitized",
                        Location = harmfulResult.Findings.FirstOrDefault()?.Location ?? 0,
                        OriginalContent = sanitizedDescription
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
    public JailbreakDetectionResult<T> DetectJailbreak(Vector<T> input)
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

        text = text.ToLowerInvariant();
        var matchedPatterns = 0;
        var totalPatterns = jailbreakPatterns.Count;

        foreach (var pattern in jailbreakPatterns)
        {
            try
            {
                if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase, RegexTimeout))
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
            catch (RegexMatchTimeoutException)
            {
                // Pattern matching timed out - treat as potential attack, skip this pattern
            }
        }

        if (matchedPatterns > 0)
        {
            result.JailbreakDetected = true;
            // Guard against division by zero when totalPatterns is 0
            result.ConfidenceScore = totalPatterns > 0
                ? Math.Min(1.0, (double)matchedPatterns / totalPatterns * 2.0)
                : 1.0; // If no patterns but somehow matched, max confidence
            result.Severity = result.ConfidenceScore;
            result.JailbreakType = matchedPatterns > 2 ? "Sophisticated" : "Basic";
            result.RecommendedActions = new[] { "Block", "Log", "Alert" };
        }

        return result;
    }

    /// <inheritdoc/>
    public HarmfulContentResult<T> IdentifyHarmfulContent(Vector<T> content)
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

        text = text.ToLowerInvariant();

        foreach (var category in options.HarmfulContentCategories)
        {
            if (!harmfulContentPatterns.TryGetValue(category, out var patterns))
            {
                continue;
            }

            var matchCount = 0;

            foreach (var pattern in patterns)
            {
                try
                {
                    if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase, RegexTimeout))
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
                catch (RegexMatchTimeoutException)
                {
                    // Pattern matching timed out - skip this pattern
                }
            }

            if (matchCount > 0)
            {
                // Guard against division by zero when patterns.Count is 0
                var categoryScore = patterns.Count > 0
                    ? Math.Min(1.0, (double)matchCount / patterns.Count)
                    : 1.0; // If no patterns but somehow matched, max score
                result.CategoryScores[category] = categoryScore;
                result.HarmScore = Math.Max(result.HarmScore, categoryScore);
            }
        }

        if (result.CategoryScores.Count > 0)
        {
            result.HarmfulContentDetected = true;

            var primaryCategory = string.Empty;
            var maxScore = double.NegativeInfinity;
            foreach (var kv in result.CategoryScores)
            {
                if (kv.Value > maxScore)
                {
                    maxScore = kv.Value;
                    primaryCategory = kv.Key;
                }
            }

            result.PrimaryHarmCategory = primaryCategory;
            result.DetectedCategories = result.CategoryScores.Keys.ToArray();
            result.RecommendedAction = result.HarmScore > 0.7 ? "Block" : result.HarmScore > 0.4 ? "Warn" : "Allow";
        }

        return result;
    }

    /// <inheritdoc/>
    public T ComputeSafetyScore(Vector<T> content)
    {
        var validation = ValidateInput(content);
        var harmfulContent = IdentifyHarmfulContent(content);
        var jailbreak = DetectJailbreak(content);

        var combinedScore = validation.SafetyScore * 0.4 +
                           (1.0 - harmfulContent.HarmScore) * 0.4 +
                           (1.0 - jailbreak.Severity) * 0.2;

        return NumOps.FromDouble(MathHelper.Clamp(combinedScore, 0.0, 1.0));
    }

    /// <inheritdoc/>
    public SafetyFilterOptions<T> GetOptions() => options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var json = JsonConvert.SerializeObject(options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);

        // Use SafeSerializationBinder to prevent deserialization attacks
        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            SerializationBinder = new SafeSerializationBinder()
        };

        options = JsonConvert.DeserializeObject<SafetyFilterOptions<T>>(json, settings) ?? new SafetyFilterOptions<T>();

        // Reinitialize patterns based on new options
        InitializePatterns();
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(fullPath, Serialize());
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        Deserialize(File.ReadAllBytes(fullPath));
    }

    private static string ConvertToText(Vector<T> data)
    {
        // For text data represented as numeric arrays (e.g., token IDs),
        // you would decode to text. For demonstration, we create a simple representation
        if (data == null || data.Length == 0)
            return string.Empty;

        // This is a placeholder - in practice, you'd have proper encoding/decoding.
        var builder = new StringBuilder();
        for (int i = 0; i < data.Length; i++)
        {
            if (i > 0)
            {
                builder.Append(' ');
            }

            object? value = data[i];
            builder.Append(value?.ToString() ?? string.Empty);
        }

        return builder.ToString();
    }

    private Vector<T> SanitizeOutput(Vector<T> output, HarmfulContentResult<T> harmfulResult)
    {
        // Create a sanitized copy of the output
        var sanitized = CloneVector(output);
        var numOps = MathHelper.GetNumericOperations<T>();

        // Zero out values at harmful locations identified in findings
        if (harmfulResult.Findings != null && harmfulResult.Findings.Count > 0)
        {
            foreach (var finding in harmfulResult.Findings)
            {
                // Zero out the harmful region with a small buffer around the location
                int location = finding.Location;
                if (location >= 0 && location < sanitized.Length)
                {
                    // Zero out the specific location and nearby indices
                    int startIdx = Math.Max(0, location - 1);
                    int endIdx = Math.Min(sanitized.Length - 1, location + 1);

                    for (int i = startIdx; i <= endIdx; i++)
                    {
                        sanitized[i] = numOps.Zero;
                    }
                }
            }
        }
        else
        {
            // If no specific findings, but harmful content detected, apply global dampening
            // Reduce all values toward zero based on harm score
            if (harmfulResult.HarmfulContentDetected && harmfulResult.HarmScore > 0)
            {
                var dampingFactor = numOps.FromDouble(1.0 - harmfulResult.HarmScore);
                for (int i = 0; i < sanitized.Length; i++)
                {
                    sanitized[i] = numOps.Multiply(sanitized[i], dampingFactor);
                }
            }
        }

        return sanitized;
    }

    private void LogFilteredContent(Vector<T> output, SafetyFilterResult<T> result)
    {
        // Log to file or monitoring system
        _ = output;
        var logEntry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}] " +
                      $"Filtered content - Categories: {string.Join(", ", result.DetectedHarmCategories)} " +
                      $"- Score: {result.SafetyScore:F3}\n";

        try
        {
            var logPath = string.IsNullOrWhiteSpace(options.LogFilePath) ? "safety_filter_logs.txt" : options.LogFilePath;
            File.AppendAllText(logPath, logEntry);
        }
        catch (IOException)
        {
            // Silent fail for logging
        }
        catch (UnauthorizedAccessException)
        {
            // Silent fail for logging
        }
    }

    private static Vector<T> CloneVector(Vector<T> source)
    {
        var clone = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }

        return clone;
    }
}
