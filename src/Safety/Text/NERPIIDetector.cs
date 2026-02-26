using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects PII using Named Entity Recognition (NER) heuristics based on contextual patterns.
/// </summary>
/// <remarks>
/// <para>
/// Goes beyond simple regex by analyzing surrounding context to identify named entities that
/// represent PII. Uses pattern-based NER with capitalization analysis, contextual cues (titles,
/// prefixes), and statistical features to detect person names, organization names, and location
/// references that regex-based approaches miss.
/// </para>
/// <para>
/// <b>For Beginners:</b> While regex can catch emails and phone numbers, it struggles with
/// person names (is "John Smith" a name or a product?). This module uses context clues — like
/// "Mr.", "CEO", "lives in" — to understand when words refer to real people, places, or
/// organizations.
/// </para>
/// <para>
/// <b>References:</b>
/// - PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)
/// - CAPID: Context-aware PII detection reducing over-redaction in QA (2026, arxiv:2602.10074)
/// - Text anonymization survey bridging NER and LLMs (2025, arxiv:2508.21587)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NERPIIDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromMilliseconds(100);

    // Contextual indicators for person names
    private static readonly string[] PersonPrefixes =
    {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sir", "lady", "lord",
        "judge", "senator", "president", "director", "officer"
    };

    private static readonly string[] PersonContextClues =
    {
        "named", "called", "known as", "my name is", "i am", "contact",
        "spoke with", "met with", "introduced", "appointed", "hired"
    };

    private static readonly string[] LocationContextClues =
    {
        "lives in", "located in", "based in", "from", "travels to",
        "address is", "resides at", "born in", "moved to", "visiting"
    };

    private static readonly string[] OrgContextClues =
    {
        "works at", "employed by", "founded", "company", "corporation",
        "organization", "inc.", "llc", "ltd", "group", "institute"
    };

    /// <inheritdoc />
    public override string ModuleName => "NERPIIDetector";

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return findings;
        }

        string lower = text.ToLowerInvariant();
        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);

        // Detect person names via context clues + capitalization
        DetectPersonNames(text, lower, words, findings);

        // Detect locations via context clues
        DetectLocations(text, lower, words, findings);

        // Detect organizations via context clues
        DetectOrganizations(text, lower, words, findings);

        return findings;
    }

    private void DetectPersonNames(string text, string lower, string[] words, List<SafetyFinding> findings)
    {
        // Check for title + capitalized name pattern
        foreach (var prefix in PersonPrefixes)
        {
            int idx = lower.IndexOf(prefix, StringComparison.Ordinal);
            while (idx >= 0)
            {
                // Look for capitalized words after the prefix
                int afterPrefix = idx + prefix.Length;
                while (afterPrefix < text.Length && text[afterPrefix] == ' ') afterPrefix++;

                if (afterPrefix < text.Length && char.IsUpper(text[afterPrefix]))
                {
                    // Extract the potential name (up to 3 capitalized words)
                    int nameEnd = afterPrefix;
                    int wordCount = 0;
                    while (nameEnd < text.Length && wordCount < 3)
                    {
                        if (text[nameEnd] == ' ')
                        {
                            if (nameEnd + 1 < text.Length && char.IsUpper(text[nameEnd + 1]))
                            {
                                wordCount++;
                                nameEnd++;
                            }
                            else break;
                        }
                        else
                        {
                            nameEnd++;
                        }
                    }

                    string name = text.Substring(afterPrefix, nameEnd - afterPrefix).Trim();
                    if (name.Length > 1)
                    {
                        findings.Add(new SafetyFinding
                        {
                            Category = SafetyCategory.PIIExposure,
                            Severity = SafetySeverity.Medium,
                            Confidence = 0.75,
                            Description = $"Potential person name detected: '{MaskPII(name)}'.",
                            RecommendedAction = SafetyAction.Warn,
                            SourceModule = ModuleName,
                            SpanStart = afterPrefix,
                            SpanEnd = nameEnd,
                            Excerpt = MaskPII(name)
                        });
                    }
                }

                idx = lower.IndexOf(prefix, afterPrefix, StringComparison.Ordinal);
            }
        }

        // Check for context clue + capitalized name pattern
        foreach (var clue in PersonContextClues)
        {
            int idx = lower.IndexOf(clue, StringComparison.Ordinal);
            if (idx >= 0)
            {
                int afterClue = idx + clue.Length;
                while (afterClue < text.Length && !char.IsLetterOrDigit(text[afterClue])) afterClue++;

                if (afterClue < text.Length && char.IsUpper(text[afterClue]))
                {
                    int nameEnd = afterClue;
                    while (nameEnd < text.Length && (char.IsLetter(text[nameEnd]) || text[nameEnd] == ' '))
                    {
                        nameEnd++;
                    }

                    string name = text.Substring(afterClue, nameEnd - afterClue).Trim();
                    if (name.Length > 1 && name.Length < 50)
                    {
                        findings.Add(new SafetyFinding
                        {
                            Category = SafetyCategory.PIIExposure,
                            Severity = SafetySeverity.Medium,
                            Confidence = 0.65,
                            Description = $"Potential person name detected near '{clue}': '{MaskPII(name)}'.",
                            RecommendedAction = SafetyAction.Warn,
                            SourceModule = ModuleName,
                            SpanStart = afterClue,
                            SpanEnd = nameEnd,
                            Excerpt = MaskPII(name)
                        });
                    }
                }
            }
        }
    }

    private void DetectLocations(string text, string lower, string[] words, List<SafetyFinding> findings)
    {
        foreach (var clue in LocationContextClues)
        {
            int idx = lower.IndexOf(clue, StringComparison.Ordinal);
            if (idx >= 0)
            {
                int afterClue = idx + clue.Length;
                while (afterClue < text.Length && !char.IsLetterOrDigit(text[afterClue])) afterClue++;

                if (afterClue < text.Length)
                {
                    int locEnd = afterClue;
                    int commaCount = 0;
                    while (locEnd < text.Length && commaCount < 2 &&
                           (char.IsLetterOrDigit(text[locEnd]) || text[locEnd] == ' ' ||
                            text[locEnd] == ',' || text[locEnd] == '.'))
                    {
                        if (text[locEnd] == ',') commaCount++;
                        locEnd++;
                    }

                    string location = text.Substring(afterClue, locEnd - afterClue).Trim().TrimEnd(',', '.');
                    if (location.Length > 2 && location.Length < 100)
                    {
                        findings.Add(new SafetyFinding
                        {
                            Category = SafetyCategory.PIIExposure,
                            Severity = SafetySeverity.Low,
                            Confidence = 0.55,
                            Description = $"Potential location/address detected near '{clue}': '{MaskPII(location)}'.",
                            RecommendedAction = SafetyAction.Log,
                            SourceModule = ModuleName,
                            SpanStart = afterClue,
                            SpanEnd = locEnd,
                            Excerpt = MaskPII(location)
                        });
                    }
                }
            }
        }
    }

    private void DetectOrganizations(string text, string lower, string[] words, List<SafetyFinding> findings)
    {
        foreach (var clue in OrgContextClues)
        {
            int idx = lower.IndexOf(clue, StringComparison.Ordinal);
            if (idx >= 0)
            {
                int afterClue = idx + clue.Length;
                while (afterClue < text.Length && !char.IsLetterOrDigit(text[afterClue])) afterClue++;

                if (afterClue < text.Length && char.IsUpper(text[afterClue]))
                {
                    int orgEnd = afterClue;
                    while (orgEnd < text.Length &&
                           (char.IsLetterOrDigit(text[orgEnd]) || text[orgEnd] == ' ' ||
                            text[orgEnd] == '.' || text[orgEnd] == '&' || text[orgEnd] == '-'))
                    {
                        orgEnd++;
                    }

                    string org = text.Substring(afterClue, orgEnd - afterClue).Trim();
                    if (org.Length > 1 && org.Length < 100)
                    {
                        findings.Add(new SafetyFinding
                        {
                            Category = SafetyCategory.PIIExposure,
                            Severity = SafetySeverity.Low,
                            Confidence = 0.50,
                            Description = $"Potential organization name detected: '{MaskPII(org)}'.",
                            RecommendedAction = SafetyAction.Log,
                            SourceModule = ModuleName,
                            SpanStart = afterClue,
                            SpanEnd = orgEnd,
                            Excerpt = MaskPII(org)
                        });
                    }
                }
            }
        }
    }

    private static string MaskPII(string text)
    {
        if (text.Length <= 2) return "**";
        return text[0] + new string('*', Math.Min(text.Length - 2, 10)) + text[text.Length - 1];
    }
}
