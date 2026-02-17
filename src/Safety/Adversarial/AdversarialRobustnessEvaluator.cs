using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Adversarial;

/// <summary>
/// Evaluates text inputs for adversarial perturbations designed to evade safety filters.
/// </summary>
/// <remarks>
/// <para>
/// Detects a variety of adversarial text manipulation techniques: homoglyph substitution
/// (replacing characters with visually similar Unicode), leetspeak encoding, zero-width
/// character insertion, text direction override attacks, and invisible character padding.
/// These techniques attempt to bypass keyword-based safety filters.
/// </para>
/// <para>
/// <b>For Beginners:</b> People sometimes try to sneak harmful content past AI safety filters
/// by using visual tricks — like replacing "a" with "а" (Cyrillic "a" that looks identical),
/// inserting invisible characters between letters, or using numbers/symbols to spell words.
/// This module catches those tricks.
/// </para>
/// <para>
/// <b>References:</b>
/// - TextFool: Fool NLP models with adversarial text (2024)
/// - Universal adversarial triggers for attacking NLP (Wallace et al., EMNLP 2019)
/// - Homoglyph attacks on content moderation (2024)
/// - Unicode security considerations (UTS #39, Unicode Consortium)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AdversarialRobustnessEvaluator<T> : ITextSafetyModule<T>
{
    private readonly double _threshold;

    // Common homoglyph mappings (Unicode characters that look like ASCII)
    private static readonly Dictionary<char, char> HomoglyphMap = new()
    {
        { '\u0430', 'a' }, // Cyrillic а
        { '\u0435', 'e' }, // Cyrillic е
        { '\u043E', 'o' }, // Cyrillic о
        { '\u0440', 'p' }, // Cyrillic р
        { '\u0441', 'c' }, // Cyrillic с
        { '\u0443', 'y' }, // Cyrillic у
        { '\u0456', 'i' }, // Cyrillic і
        { '\u0445', 'x' }, // Cyrillic х
        { '\u04BB', 'h' }, // Cyrillic һ
        { '\u0261', 'g' }, // Latin g
        { '\uFF41', 'a' }, // Fullwidth a
        { '\uFF45', 'e' }, // Fullwidth e
        { '\uFF4F', 'o' }, // Fullwidth o
        { '\u0391', 'A' }, // Greek Α
        { '\u0392', 'B' }, // Greek Β
        { '\u0395', 'E' }, // Greek Ε
        { '\u0397', 'H' }, // Greek Η
        { '\u039A', 'K' }, // Greek Κ
        { '\u039C', 'M' }, // Greek Μ
        { '\u039D', 'N' }, // Greek Ν
        { '\u039F', 'O' }, // Greek Ο
        { '\u03A1', 'P' }, // Greek Ρ
        { '\u03A4', 'T' }, // Greek Τ
    };

    // Zero-width and invisible characters
    private static readonly HashSet<char> InvisibleChars = new()
    {
        '\u200B', // Zero-width space
        '\u200C', // Zero-width non-joiner
        '\u200D', // Zero-width joiner
        '\u200E', // Left-to-right mark
        '\u200F', // Right-to-left mark
        '\u202A', // Left-to-right embedding
        '\u202B', // Right-to-left embedding
        '\u202C', // Pop directional formatting
        '\u202D', // Left-to-right override
        '\u202E', // Right-to-left override
        '\u2060', // Word joiner
        '\u2061', // Function application
        '\u2062', // Invisible times
        '\u2063', // Invisible separator
        '\u2064', // Invisible plus
        '\uFEFF', // Zero-width no-break space (BOM)
    };

    // Leetspeak mappings
    private static readonly Dictionary<char, char> LeetspeakMap = new()
    {
        { '0', 'o' }, { '1', 'l' }, { '3', 'e' }, { '4', 'a' },
        { '5', 's' }, { '7', 't' }, { '8', 'b' }, { '@', 'a' },
        { '$', 's' }, { '!', 'i' }, { '+', 't' },
    };

    /// <inheritdoc />
    public string ModuleName => "AdversarialRobustnessEvaluator";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new adversarial robustness evaluator.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.4.</param>
    public AdversarialRobustnessEvaluator(double threshold = 0.4)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text)) return findings;

        // 1. Homoglyph detection
        double homoglyphScore = DetectHomoglyphs(text);
        if (homoglyphScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.High,
                Confidence = Math.Min(1.0, homoglyphScore),
                Description = $"Homoglyph attack detected (score: {homoglyphScore:F3}). " +
                              $"Text contains visually similar Unicode characters replacing ASCII " +
                              $"(e.g., Cyrillic 'а' for Latin 'a'). This is often used to bypass filters.",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }

        // 2. Invisible character detection
        double invisibleScore = DetectInvisibleCharacters(text);
        if (invisibleScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.High,
                Confidence = Math.Min(1.0, invisibleScore),
                Description = $"Invisible character injection detected (score: {invisibleScore:F3}). " +
                              $"Text contains zero-width or control characters that may be used " +
                              $"to evade content filters or hide malicious content.",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }

        // 3. Leetspeak/obfuscation detection
        double leetspeakScore = DetectLeetspeak(text);
        if (leetspeakScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Medium,
                Confidence = Math.Min(1.0, leetspeakScore),
                Description = $"Text obfuscation detected (score: {leetspeakScore:F3}). " +
                              $"Text uses character substitutions (leetspeak) that may attempt " +
                              $"to bypass keyword-based safety filters.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        // 4. Mixed-script detection
        double mixedScriptScore = DetectMixedScripts(text);
        if (mixedScriptScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Medium,
                Confidence = Math.Min(1.0, mixedScriptScore),
                Description = $"Suspicious mixed-script text detected (score: {mixedScriptScore:F3}). " +
                              $"Individual words contain characters from multiple Unicode scripts, " +
                              $"which may indicate a homoglyph-based evasion attempt.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return Array.Empty<SafetyFinding>();
    }

    private static double DetectHomoglyphs(string text)
    {
        int homoglyphCount = 0;
        int totalChars = 0;

        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c) || char.IsControl(c)) continue;
            totalChars++;
            if (HomoglyphMap.ContainsKey(c)) homoglyphCount++;
        }

        if (totalChars == 0) return 0;

        // Even a few homoglyphs in otherwise ASCII text is suspicious
        double ratio = (double)homoglyphCount / totalChars;
        if (homoglyphCount == 0) return 0;
        if (homoglyphCount >= 3) return Math.Min(1.0, 0.5 + ratio * 5);
        return Math.Min(1.0, ratio * 10);
    }

    private static double DetectInvisibleCharacters(string text)
    {
        int invisibleCount = 0;
        int totalChars = text.Length;

        foreach (char c in text)
        {
            if (InvisibleChars.Contains(c)) invisibleCount++;
        }

        if (totalChars == 0 || invisibleCount == 0) return 0;

        // Any invisible characters in user input are suspicious
        if (invisibleCount >= 5) return 1.0;
        if (invisibleCount >= 2) return 0.8;
        return 0.5;
    }

    private static double DetectLeetspeak(string text)
    {
        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);

        int leetspeakWords = 0;
        int totalWords = 0;

        foreach (string word in words)
        {
            if (word.Length < 3) continue;
            totalWords++;

            // Check if word contains leetspeak substitutions mixed with letters
            bool hasLetter = false;
            int substitutionCount = 0;

            foreach (char c in word)
            {
                if (char.IsLetter(c)) hasLetter = true;
                if (LeetspeakMap.ContainsKey(c)) substitutionCount++;
            }

            if (hasLetter && substitutionCount >= 1 && substitutionCount <= word.Length - 1)
            {
                leetspeakWords++;
            }
        }

        if (totalWords == 0) return 0;

        double ratio = (double)leetspeakWords / totalWords;
        // A few leetspeak words is unusual; many is very suspicious
        return Math.Min(1.0, ratio * 3);
    }

    private static double DetectMixedScripts(string text)
    {
        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);

        int mixedScriptWords = 0;
        int totalWords = 0;

        foreach (string word in words)
        {
            if (word.Length < 3) continue;
            totalWords++;

            bool hasLatin = false, hasCyrillic = false, hasGreek = false;

            foreach (char c in word)
            {
                if (c >= 'A' && c <= 'z') hasLatin = true;
                else if (c >= '\u0400' && c <= '\u04FF') hasCyrillic = true;
                else if (c >= '\u0370' && c <= '\u03FF') hasGreek = true;
            }

            int scriptCount = (hasLatin ? 1 : 0) + (hasCyrillic ? 1 : 0) + (hasGreek ? 1 : 0);
            if (scriptCount >= 2) mixedScriptWords++;
        }

        if (totalWords == 0) return 0;

        double ratio = (double)mixedScriptWords / totalWords;
        if (mixedScriptWords == 0) return 0;
        if (mixedScriptWords >= 3) return Math.Min(1.0, 0.7 + ratio);
        return Math.Min(1.0, ratio * 5);
    }
}
