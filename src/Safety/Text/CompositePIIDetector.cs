using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Combines multiple PII detection strategies into a unified detector with deduplication.
/// </summary>
/// <remarks>
/// <para>
/// Runs multiple PII detectors (regex, NER, context-aware) and merges their results with
/// span-level deduplication. When multiple detectors flag the same text region, the finding
/// with the highest confidence is kept. This provides comprehensive coverage while avoiding
/// duplicate alerts.
/// </para>
/// <para>
/// <b>For Beginners:</b> Different PII detection methods are good at finding different types
/// of personal information. This module runs all of them and combines the results, removing
/// duplicates so you get one clean list of all PII found.
/// </para>
/// <para>
/// <b>References:</b>
/// - Hybrid multilingual PII detection (2025, arxiv:2510.07551)
/// - PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CompositePIIDetector<T> : TextSafetyModuleBase<T>
{
    private readonly ITextSafetyModule<T>[] _detectors;

    /// <inheritdoc />
    public override string ModuleName => "CompositePIIDetector";

    /// <summary>
    /// Initializes a new composite PII detector with default sub-detectors.
    /// </summary>
    public CompositePIIDetector()
    {
        _detectors = new ITextSafetyModule<T>[]
        {
            new ContextAwarePIIDetector<T>(new RegexPIIDetector<T>()),
            new NERPIIDetector<T>()
        };
    }

    /// <summary>
    /// Initializes a new composite PII detector with custom sub-detectors.
    /// </summary>
    /// <param name="detectors">The PII detectors to combine.</param>
    public CompositePIIDetector(ITextSafetyModule<T>[] detectors)
    {
        _detectors = detectors ?? throw new ArgumentNullException(nameof(detectors));
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<SafetyFinding>();
        }

        // Collect all findings from all detectors
        var allFindings = new List<SafetyFinding>();
        foreach (var detector in _detectors)
        {
            allFindings.AddRange(detector.EvaluateText(text));
        }

        // Deduplicate by span overlap — keep highest confidence for overlapping regions
        return DeduplicateFindings(allFindings);
    }

    private static IReadOnlyList<SafetyFinding> DeduplicateFindings(List<SafetyFinding> findings)
    {
        if (findings.Count <= 1) return findings;

        // Sort by confidence descending
        findings.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

        var result = new List<SafetyFinding>();
        var coveredRanges = new List<(int start, int end)>();

        foreach (var finding in findings)
        {
            int start = finding.SpanStart;
            int end = finding.SpanEnd;

            // If no span info, always include (can't deduplicate)
            if (start < 0 || end < 0)
            {
                // Check if same category and description already exists
                bool duplicate = false;
                foreach (var existing in result)
                {
                    if (existing.Category == finding.Category &&
                        existing.Description == finding.Description)
                    {
                        duplicate = true;
                        break;
                    }
                }

                if (!duplicate)
                {
                    result.Add(finding);
                }
                continue;
            }

            // Check for overlap with already-covered ranges
            bool overlaps = false;
            foreach (var (rStart, rEnd) in coveredRanges)
            {
                // Check if ranges overlap significantly (>50% of either range)
                int overlapStart = Math.Max(start, rStart);
                int overlapEnd = Math.Min(end, rEnd);
                int overlapLength = Math.Max(0, overlapEnd - overlapStart);

                int findingLength = end - start;
                int existingLength = rEnd - rStart;

                if (findingLength > 0 && existingLength > 0)
                {
                    double overlapRatio = (double)overlapLength / Math.Min(findingLength, existingLength);
                    if (overlapRatio > 0.5)
                    {
                        overlaps = true;
                        break;
                    }
                }
            }

            if (!overlaps)
            {
                result.Add(finding);
                coveredRanges.Add((start, end));
            }
        }

        return result;
    }
}
