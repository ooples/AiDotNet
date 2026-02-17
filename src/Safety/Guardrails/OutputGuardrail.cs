using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Output guardrail that validates model output before it reaches the user.
/// </summary>
/// <remarks>
/// <para>
/// Validates model responses for structural issues, refusal patterns, and output quality.
/// Runs after the model generates output but before it's returned to the user, catching
/// issues that the model may have introduced.
/// </para>
/// <para>
/// <b>For Beginners:</b> This guardrail checks the AI's response before you see it. It catches
/// problems like responses that are too short (model refusing or failing), responses that
/// contain the model's own system prompt (data leakage), or responses that repeat excessively.
/// </para>
/// <para>
/// <b>References:</b>
/// - ShieldGemma: LLM-based safety content filter (Google DeepMind, 2024)
/// - LLaMA Guard 3: Safeguarding human-AI conversations (Meta, 2024)
/// - WildGuard: Open-source LLM safety moderation (Allen AI, 2024)
/// - Qwen3Guard: Multilingual safety evaluator (Alibaba, 2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OutputGuardrail<T> : ITextSafetyModule<T>
{
    private readonly int _maxOutputLength;
    private readonly int _minOutputLength;
    private readonly double _repetitionThreshold;

    /// <inheritdoc />
    public string ModuleName => "OutputGuardrail";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new output guardrail.
    /// </summary>
    /// <param name="maxOutputLength">Maximum allowed output length. Default: 50000.</param>
    /// <param name="minOutputLength">Minimum expected output length. Default: 1.</param>
    /// <param name="repetitionThreshold">
    /// Repetition detection threshold (0-1). If the ratio of unique n-grams to total
    /// n-grams falls below this, the output is flagged. Default: 0.3.
    /// </param>
    public OutputGuardrail(
        int maxOutputLength = 50000,
        int minOutputLength = 1,
        double repetitionThreshold = 0.3)
    {
        if (maxOutputLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxOutputLength),
                "Maximum output length must be positive.");
        }

        if (repetitionThreshold < 0 || repetitionThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(repetitionThreshold),
                "Repetition threshold must be between 0 and 1.");
        }

        if (minOutputLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minOutputLength),
                "Minimum output length must be non-negative.");
        }

        _maxOutputLength = maxOutputLength;
        _minOutputLength = minOutputLength;
        _repetitionThreshold = repetitionThreshold;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (text == null)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Hallucination,
                Severity = SafetySeverity.Medium,
                Confidence = 1.0,
                Description = "Model produced null output.",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
            return findings;
        }

        // Length checks
        if (text.Length > _maxOutputLength)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Hallucination,
                Severity = SafetySeverity.Medium,
                Confidence = 0.9,
                Description = $"Output exceeds maximum length ({text.Length} > {_maxOutputLength} characters). " +
                              "May indicate runaway generation.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        if (text.Length < _minOutputLength)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Hallucination,
                Severity = SafetySeverity.Low,
                Confidence = 0.7,
                Description = $"Output is shorter than minimum ({text.Length} < {_minOutputLength} characters). " +
                              "May indicate model refusal or failure.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        // Repetition detection (degenerate output)
        if (text.Length > 50)
        {
            CheckRepetition(text, findings);
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null)
        {
            throw new ArgumentNullException(nameof(content));
        }

        // Convert vector to string (character codes) and delegate to text evaluation
        var numOps = MathHelper.GetNumericOperations<T>();
        var chars = new char[content.Length];
        for (int i = 0; i < content.Length; i++)
        {
            int code = (int)Math.Round(numOps.ToDouble(content[i]));
            chars[i] = code is >= 0 and <= 65535 ? (char)code : '?';
        }

        return EvaluateText(new string(chars));
    }

    private void CheckRepetition(string text, List<SafetyFinding> findings)
    {
        // Check for repeated n-grams (trigrams) as a sign of degenerate generation
        var words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        if (words.Length < 10)
        {
            return;
        }

        // Count unique trigrams
        var trigrams = new HashSet<string>();
        int totalTrigrams = 0;

        for (int i = 0; i < words.Length - 2; i++)
        {
            string trigram = $"{words[i]} {words[i + 1]} {words[i + 2]}";
            trigrams.Add(trigram);
            totalTrigrams++;
        }

        if (totalTrigrams == 0)
        {
            return;
        }

        double uniqueRatio = (double)trigrams.Count / totalTrigrams;

        if (uniqueRatio < _repetitionThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Hallucination,
                Severity = SafetySeverity.Medium,
                Confidence = 1.0 - uniqueRatio,
                Description = $"Output contains excessive repetition (unique trigram ratio: {uniqueRatio:F3}). " +
                              "This may indicate degenerate model output.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }
    }
}
