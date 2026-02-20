using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Input guardrail that validates user input before it reaches the model.
/// </summary>
/// <remarks>
/// <para>
/// Enforces structural input constraints including maximum length, required format,
/// and basic sanity checks. Acts as the first line of defense before content-specific
/// safety modules run.
/// </para>
/// <para>
/// <b>For Beginners:</b> This guardrail checks user input before it's processed by the AI.
/// It catches problems like inputs that are too long, empty inputs, or inputs that contain
/// suspicious patterns (like prompt injection attempts using special characters).
/// </para>
/// <para>
/// <b>References:</b>
/// - NeMo Guardrails: Input/output rails for LLM applications (NVIDIA, 2024)
/// - Guardrails AI: Production-grade validation framework (2024)
/// - LLM Guard: Input validation for production systems (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InputGuardrail<T> : ITextSafetyModule<T>
{
    private readonly int _maxInputLength;
    private readonly bool _blockEmptyInput;
    private readonly bool _detectSpecialTokenInjection;

    /// <inheritdoc />
    public string ModuleName => "InputGuardrail";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new input guardrail.
    /// </summary>
    /// <param name="maxInputLength">Maximum allowed input length in characters. Default: 10000.</param>
    /// <param name="blockEmptyInput">Whether to block empty/whitespace-only input. Default: true.</param>
    /// <param name="detectSpecialTokenInjection">
    /// Whether to detect special token injection attempts (control characters,
    /// zero-width characters, etc.). Default: true.
    /// </param>
    public InputGuardrail(
        int maxInputLength = 10000,
        bool blockEmptyInput = true,
        bool detectSpecialTokenInjection = true)
    {
        if (maxInputLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxInputLength),
                "Maximum input length must be positive.");
        }

        _maxInputLength = maxInputLength;
        _blockEmptyInput = blockEmptyInput;
        _detectSpecialTokenInjection = detectSpecialTokenInjection;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        // Null input: always return empty (no findings to report on null)
        if (text is null)
        {
            return findings;
        }

        // Empty input check
        if (_blockEmptyInput && string.IsNullOrWhiteSpace(text))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Low,
                Confidence = 1.0,
                Description = "Input is empty or contains only whitespace.",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
            return findings;
        }

        // Length check
        if (text.Length > _maxInputLength)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Medium,
                Confidence = 1.0,
                Description = $"Input exceeds maximum length ({text.Length} > {_maxInputLength} characters).",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }

        // Special token injection detection
        if (_detectSpecialTokenInjection)
        {
            CheckSpecialTokenInjection(text, findings);
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        // For vector input, only check length
        var findings = new List<SafetyFinding>();

        if (content.Length == 0 && _blockEmptyInput)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Low,
                Confidence = 1.0,
                Description = "Input vector is empty.",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private void CheckSpecialTokenInjection(string text, List<SafetyFinding> findings)
    {
        int controlCharCount = 0;
        int zeroWidthCount = 0;

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];

            // Control characters (excluding common whitespace)
            if (char.IsControl(c) && c != '\n' && c != '\r' && c != '\t')
            {
                controlCharCount++;
            }

            // Zero-width characters (used for Unicode tag smuggling)
            // Note: Unicode tag characters U+E0001-U+E007F are supplementary plane characters
            // that require surrogate pair detection, handled separately below.
            if (c == '\u200B' || c == '\u200C' || c == '\u200D' || c == '\uFEFF')
            {
                zeroWidthCount++;
            }
        }

        // Check for Unicode tag characters (supplementary plane U+E0001-U+E007F)
        for (int i = 0; i < text.Length - 1; i++)
        {
            if (char.IsHighSurrogate(text[i]) && char.IsLowSurrogate(text[i + 1]))
            {
                int codePoint = char.ConvertToUtf32(text[i], text[i + 1]);
                if (codePoint >= 0xE0001 && codePoint <= 0xE007F)
                {
                    zeroWidthCount++;
                }
            }
        }

        if (controlCharCount > 0)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Medium,
                Confidence = 0.8,
                Description = $"Input contains {controlCharCount} control character(s) that may indicate " +
                              "special token injection or encoding attacks.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        if (zeroWidthCount > 3)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.High,
                Confidence = 0.9,
                Description = $"Input contains {zeroWidthCount} zero-width characters that may indicate " +
                              "Unicode tag smuggling or hidden content injection.",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }
    }
}
