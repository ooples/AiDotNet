using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Components;

/// <summary>
/// Samples diverse thoughts to avoid redundant exploration of similar reasoning paths.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Diversity sampling ensures you explore different types of solutions
/// rather than repeatedly trying similar approaches. It's like brainstorming rules: instead of
/// listing "apple, orange, banana, grape" (all fruits), you want "apple, carrot, bread, milk"
/// (diverse food categories).
///
/// **Why it matters:**
/// - Prevents wasting compute on similar reasoning paths
/// - Ensures comprehensive exploration of solution space
/// - Increases chances of finding creative/unexpected solutions
/// - Critical for complex problems with multiple valid approaches
///
/// **How it works:**
/// 1. Calculate diversity scores between all candidate thoughts
/// 2. Select thoughts that are maximally different from each other
/// 3. Use greedy selection or clustering algorithms
///
/// **Example:**
/// Given candidates for "reduce carbon emissions":
/// - "Use solar panels" (energy)
/// - "Use wind turbines" (energy - SIMILAR to solar)
/// - "Electric vehicles" (transportation - DIFFERENT)
/// - "Hybrid cars" (transportation - SIMILAR to EV)
/// - "Carbon capture" (industrial - DIFFERENT)
///
/// Diverse sample (N=3): [solar panels, electric vehicles, carbon capture]
/// Non-diverse sample (N=3): [solar panels, wind turbines, electric vehicles] - too energy-focused
/// </para>
/// </remarks>
public class DiversitySampler<T> : IDiversitySampler<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="DiversitySampler{T}"/> class.
    /// </summary>
    public DiversitySampler()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public List<AiDotNet.Reasoning.Models.ThoughtNode<T>> SampleDiverse(
        List<AiDotNet.Reasoning.Models.ThoughtNode<T>> candidates,
        int numToSample,
        ReasoningConfig config)
    {
        if (candidates == null || candidates.Count == 0)
            throw new ArgumentException("Candidates list cannot be null or empty", nameof(candidates));

        if (numToSample < 1)
            throw new ArgumentException("Must sample at least 1 thought", nameof(numToSample));

        // If requesting more than available, return all
        if (numToSample >= candidates.Count)
        {
            return new List<AiDotNet.Reasoning.Models.ThoughtNode<T>>(candidates);
        }

        // Use greedy selection for diversity
        var selected = new List<AiDotNet.Reasoning.Models.ThoughtNode<T>>();

        // Start with the highest-scoring candidate
        var firstNode = candidates.OrderByDescending(c => Convert.ToDouble(c.EvaluationScore)).First();
        selected.Add(firstNode);

        // Greedily add candidates that maximize diversity from already-selected
        while (selected.Count < numToSample)
        {
            AiDotNet.Reasoning.Models.ThoughtNode<T>? bestCandidate = null;
            double maxMinDiversity = double.MinValue;

            foreach (var candidate in candidates)
            {
                // Skip if already selected
                if (selected.Contains(candidate))
                    continue;

                // Calculate minimum diversity to any selected node
                double minDiversity = double.MaxValue;
                foreach (var selectedNode in selected)
                {
                    T diversity = CalculateDiversity(candidate, selectedNode);
                    double diversityValue = Convert.ToDouble(diversity);

                    if (diversityValue < minDiversity)
                    {
                        minDiversity = diversityValue;
                    }
                }

                // Select the candidate with the highest minimum diversity
                // (i.e., the one that's most different from all selected nodes)
                if (minDiversity > maxMinDiversity)
                {
                    maxMinDiversity = minDiversity;
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate != null)
            {
                selected.Add(bestCandidate);
            }
            else
            {
                // No more candidates available
                break;
            }
        }

        return selected;
    }

    /// <inheritdoc/>
    public T CalculateDiversity(AiDotNet.Reasoning.Models.ThoughtNode<T> thought1, AiDotNet.Reasoning.Models.ThoughtNode<T> thought2)
    {
        if (thought1 == null || thought2 == null)
            return _numOps.Zero;

        // Calculate diversity based on text similarity
        // Higher diversity score = more different

        string text1 = thought1.Thought.ToLowerInvariant();
        string text2 = thought2.Thought.ToLowerInvariant();

        // Method 1: Simple word overlap (Jaccard distance)
        var words1 = ExtractWords(text1);
        var words2 = ExtractWords(text2);

        if (words1.Count == 0 && words2.Count == 0)
            return _numOps.Zero;

        if (words1.Count == 0 || words2.Count == 0)
            return _numOps.One;

        // Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        int intersection = words1.Intersect(words2).Count();
        int union = words1.Union(words2).Count();

        double similarity = union > 0 ? (double)intersection / union : 0.0;

        // Diversity is 1 - similarity
        double diversity = 1.0 - similarity;

        // Boost diversity if thoughts belong to different categories/domains
        if (BelongToDifferentDomains(text1, text2))
        {
            diversity = Math.Min(1.0, diversity * 1.3); // 30% boost
        }

        return _numOps.FromDouble(diversity);
    }

    /// <summary>
    /// Extracts significant words from text (excluding common stop words).
    /// </summary>
    private HashSet<string> ExtractWords(string text)
    {
        // Simple tokenization
        var words = text.Split(new[] { ' ', ',', '.', '!', '?', ';', ':', '\n', '\t' },
            StringSplitOptions.RemoveEmptyEntries);

        // Remove common stop words
        var stopWords = new HashSet<string>
        {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "this", "that", "these", "those", "it", "its", "we", "our"
        };

        var significantWords = new HashSet<string>();
        foreach (var word in words)
        {
            string cleaned = word.Trim().ToLowerInvariant();
            if (cleaned.Length > 2 && !stopWords.Contains(cleaned))
            {
                significantWords.Add(cleaned);
            }
        }

        return significantWords;
    }

    /// <summary>
    /// Checks if two thoughts belong to different domains or categories.
    /// </summary>
    private bool BelongToDifferentDomains(string text1, string text2)
    {
        // Define domain keywords
        var domainKeywords = new Dictionary<string, HashSet<string>>
        {
            ["energy"] = new() { "solar", "wind", "renewable", "power", "electricity", "nuclear", "fossil" },
            ["transportation"] = new() { "vehicle", "car", "train", "bus", "bike", "transport", "traffic" },
            ["industrial"] = new() { "factory", "manufacturing", "production", "industry", "process", "facility" },
            ["agriculture"] = new() { "farm", "crop", "livestock", "agriculture", "food", "land", "soil" },
            ["technology"] = new() { "software", "algorithm", "data", "computer", "digital", "ai", "technology" },
            ["policy"] = new() { "regulation", "law", "policy", "government", "tax", "legislation" }
        };

        // Determine domains for each text
        var domains1 = new HashSet<string>();
        var domains2 = new HashSet<string>();

        foreach (var (domain, keywords) in domainKeywords)
        {
            if (keywords.Any(k => text1.Contains(k)))
            {
                domains1.Add(domain);
            }

            if (keywords.Any(k => text2.Contains(k)))
            {
                domains2.Add(domain);
            }
        }

        // Different domains if there's no overlap
        return domains1.Count > 0 && domains2.Count > 0 && !domains1.Overlaps(domains2);
    }
}
