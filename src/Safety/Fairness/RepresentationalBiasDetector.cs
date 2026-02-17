using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Fairness;

/// <summary>
/// Detects representational bias by analyzing whether demographic groups are underrepresented,
/// overrepresented, or systematically associated with specific roles/contexts in text.
/// </summary>
/// <remarks>
/// <para>
/// Representational bias occurs when certain groups are disproportionately represented in
/// specific contexts. For example, if "doctor" almost always appears near male pronouns
/// and "nurse" near female pronouns, the text reinforces occupational representation bias.
/// This module counts demographic group mentions and measures role-association imbalances.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine a children's book where all the scientists are men and
/// all the teachers are women. Even if nothing negative is said, the lopsided representation
/// teaches children that certain roles "belong" to certain groups. This detector finds
/// exactly that kind of imbalance in AI-generated text.
/// </para>
/// <para>
/// <b>References:</b>
/// - BEATS: Comprehensive bias evaluation test suite for LLMs (2025, arxiv:2503.24310)
/// - Representation bias in text generation (Sheng et al., EMNLP 2019)
/// - Measuring representational harms in language technology (Blodgett et al., 2020)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class RepresentationalBiasDetector<T> : ITextSafetyModule<T>
{
    private readonly double _disparityThreshold;
    private readonly string[] _protectedAttributes;

    // Role/context categories with associated terms
    private static readonly Dictionary<string, string[]> RoleCategories = new(StringComparer.OrdinalIgnoreCase)
    {
        ["leadership"] = new[] { "ceo", "president", "director", "manager", "executive", "boss", "chief", "leader", "chairman", "founder" },
        ["stem"] = new[] { "scientist", "engineer", "programmer", "developer", "physicist", "mathematician", "researcher", "technologist", "analyst", "architect" },
        ["caregiving"] = new[] { "nurse", "teacher", "caregiver", "nanny", "babysitter", "daycare", "aide", "assistant", "receptionist", "secretary" },
        ["service"] = new[] { "cleaner", "janitor", "maid", "servant", "waiter", "waitress", "barista", "cashier", "housekeeper" },
        ["creative"] = new[] { "artist", "writer", "musician", "designer", "poet", "actor", "singer", "dancer", "filmmaker" },
        ["authority"] = new[] { "judge", "lawyer", "officer", "detective", "commander", "general", "senator", "governor", "politician" },
        ["criminal"] = new[] { "criminal", "thief", "suspect", "inmate", "prisoner", "convict", "offender", "gangster", "dealer" },
        ["victim"] = new[] { "victim", "refugee", "homeless", "poor", "helpless", "vulnerable", "dependent", "needy" }
    };

    // Demographic group terms by attribute
    private static readonly Dictionary<string, string[][]> DemographicGroups = new(StringComparer.OrdinalIgnoreCase)
    {
        ["gender"] = new[]
        {
            new[] { "man", "men", "male", "boy", "he", "him", "his", "father", "husband", "brother", "son" },
            new[] { "woman", "women", "female", "girl", "she", "her", "hers", "mother", "wife", "sister", "daughter" }
        },
        ["race"] = new[]
        {
            new[] { "white", "caucasian", "european" },
            new[] { "black", "african", "african american" },
            new[] { "asian", "east asian", "south asian" },
            new[] { "hispanic", "latino", "latina", "latinx" }
        },
        ["age"] = new[]
        {
            new[] { "young", "youth", "teenager", "millennial", "gen z", "child", "kid" },
            new[] { "old", "elderly", "senior", "aged", "retired", "boomer" }
        }
    };

    /// <inheritdoc />
    public string ModuleName => "RepresentationalBiasDetector";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new representational bias detector.
    /// </summary>
    /// <param name="disparityThreshold">
    /// Maximum allowed representation disparity between groups (0.0-1.0).
    /// A value of 0.3 means one group can appear at most 30% more frequently in a role. Default: 0.3.
    /// </param>
    /// <param name="protectedAttributes">Attribute categories to analyze. Default: ["gender", "race"].</param>
    public RepresentationalBiasDetector(
        double disparityThreshold = 0.3,
        string[]? protectedAttributes = null)
    {
        _disparityThreshold = disparityThreshold;
        _protectedAttributes = protectedAttributes ?? new[] { "gender", "race" };
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text) || text.Length < 30) return findings;

        string lower = text.ToLowerInvariant();
        string[] words = lower.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?', '(', ')' },
            StringSplitOptions.RemoveEmptyEntries);

        foreach (string attribute in _protectedAttributes)
        {
            if (!DemographicGroups.TryGetValue(attribute, out var groups)) continue;

            // Count total mentions of each group
            int[] groupMentionCounts = new int[groups.Length];
            for (int g = 0; g < groups.Length; g++)
            {
                foreach (string word in words)
                {
                    foreach (string term in groups[g])
                    {
                        if (word == term || word.Contains(term))
                        {
                            groupMentionCounts[g]++;
                            break;
                        }
                    }
                }
            }

            // Check mention count disparity (underrepresentation)
            int totalMentions = 0;
            foreach (int c in groupMentionCounts) totalMentions += c;
            if (totalMentions < 4) continue;

            int mentionedGroups = 0;
            foreach (int c in groupMentionCounts)
            {
                if (c > 0) mentionedGroups++;
            }

            if (mentionedGroups < 2) continue;

            double expectedShare = 1.0 / mentionedGroups;
            double maxMentionDisparity = 0;
            string overrepresented = "";
            string underrepresented = "";

            for (int g = 0; g < groups.Length; g++)
            {
                if (groupMentionCounts[g] == 0) continue;
                double actualShare = (double)groupMentionCounts[g] / totalMentions;
                double disparity = Math.Abs(actualShare - expectedShare);
                if (disparity > maxMentionDisparity)
                {
                    maxMentionDisparity = disparity;
                    if (actualShare > expectedShare)
                    {
                        overrepresented = groups[g][0];
                        // Find the most underrepresented
                        double minShare = 1.0;
                        for (int g2 = 0; g2 < groups.Length; g2++)
                        {
                            if (groupMentionCounts[g2] > 0)
                            {
                                double s2 = (double)groupMentionCounts[g2] / totalMentions;
                                if (s2 < minShare) { minShare = s2; underrepresented = groups[g2][0]; }
                            }
                        }
                    }
                }
            }

            // Check role-association bias
            foreach (var roleEntry in RoleCategories)
            {
                string roleName = roleEntry.Key;
                string[] roleTerms = roleEntry.Value;

                var roleGroupCounts = new int[groups.Length];
                int totalRoleAssociations = 0;

                for (int g = 0; g < groups.Length; g++)
                {
                    roleGroupCounts[g] = CountProximityAssociations(words, groups[g], roleTerms, 12);
                    totalRoleAssociations += roleGroupCounts[g];
                }

                if (totalRoleAssociations < 2) continue;

                // Find disparity in role association
                double maxRoleDisparity = 0;
                string roleOverGroup = "";
                string roleUnderGroup = "";

                for (int i = 0; i < groups.Length; i++)
                {
                    if (roleGroupCounts[i] == 0) continue;
                    for (int j = i + 1; j < groups.Length; j++)
                    {
                        if (roleGroupCounts[j] == 0) continue;
                        double shareI = (double)roleGroupCounts[i] / totalRoleAssociations;
                        double shareJ = (double)roleGroupCounts[j] / totalRoleAssociations;
                        double disparity = Math.Abs(shareI - shareJ);
                        if (disparity > maxRoleDisparity)
                        {
                            maxRoleDisparity = disparity;
                            roleOverGroup = shareI > shareJ ? groups[i][0] : groups[j][0];
                            roleUnderGroup = shareI > shareJ ? groups[j][0] : groups[i][0];
                        }
                    }
                }

                if (maxRoleDisparity >= _disparityThreshold)
                {
                    findings.Add(new SafetyFinding
                    {
                        Category = SafetyCategory.Bias,
                        Severity = maxRoleDisparity >= 0.6 ? SafetySeverity.High : SafetySeverity.Medium,
                        Confidence = Math.Min(1.0, maxRoleDisparity),
                        Description = $"Representational bias in '{roleName}' roles for '{attribute}': " +
                                      $"'{roleOverGroup}' is {maxRoleDisparity * 100:F0}% more associated " +
                                      $"with {roleName} roles than '{roleUnderGroup}'.",
                        RecommendedAction = SafetyAction.Warn,
                        SourceModule = ModuleName
                    });
                }
            }

            // Overall mention imbalance finding
            if (maxMentionDisparity >= _disparityThreshold && !string.IsNullOrEmpty(overrepresented))
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Bias,
                    Severity = SafetySeverity.Low,
                    Confidence = Math.Min(1.0, maxMentionDisparity),
                    Description = $"Representational imbalance for '{attribute}': " +
                                  $"'{overrepresented}' is overrepresented compared to '{underrepresented}' " +
                                  $"(disparity: {maxMentionDisparity:F3}).",
                    RecommendedAction = SafetyAction.Log,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return Array.Empty<SafetyFinding>();
    }

    private static int CountProximityAssociations(string[] words, string[] groupTerms, string[] roleTerms, int windowSize)
    {
        int count = 0;
        for (int i = 0; i < words.Length; i++)
        {
            bool isGroupTerm = false;
            foreach (string term in groupTerms)
            {
                if (words[i] == term || words[i].Contains(term))
                {
                    isGroupTerm = true;
                    break;
                }
            }
            if (!isGroupTerm) continue;

            int start = Math.Max(0, i - windowSize);
            int end = Math.Min(words.Length, i + windowSize + 1);
            for (int j = start; j < end; j++)
            {
                if (j == i) continue;
                foreach (string role in roleTerms)
                {
                    if (words[j] == role || words[j].Contains(role))
                    {
                        count++;
                        break;
                    }
                }
            }
        }
        return count;
    }
}
