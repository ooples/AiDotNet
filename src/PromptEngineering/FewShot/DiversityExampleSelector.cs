using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.FewShot;

/// <summary>
/// Selects diverse examples to maximize coverage of different patterns.
/// </summary>
/// <typeparam name="T">The type of numeric data used for similarity scoring.</typeparam>
/// <remarks>
/// <para>
/// This selector picks examples that are different from each other to provide broad coverage.
/// It uses a greedy algorithm to iteratively select examples that are most different from
/// already-selected examples.
/// </para>
/// <para><b>For Beginners:</b> Picks examples that are different from each other.
///
/// Instead of picking similar examples, this ensures variety:
/// - If you have 100 sentiment examples
/// - Random might give 3 positive examples
/// - Diversity gives 1 positive, 1 negative, 1 neutral
///
/// Example:
/// <code>
/// var selector = new DiversityExampleSelector&lt;double&gt;(embeddingFunction);
///
/// // Add many sentiment examples
/// selector.AddExample(new FewShotExample { Input = "Great!", Output = "Positive" });
/// selector.AddExample(new FewShotExample { Input = "Terrible!", Output = "Negative" });
/// selector.AddExample(new FewShotExample { Input = "It's okay", Output = "Neutral" });
/// // ... more examples
///
/// // Get diverse examples
/// var examples = selector.SelectExamples("Review this product", 3);
/// // Returns one from each sentiment category
/// </code>
///
/// Use this when:
/// - You want broad coverage of the example space
/// - Your examples cluster into natural groups
/// - Model needs to understand the full range of possibilities
/// </para>
/// </remarks>
public class DiversityExampleSelector<T> : FewShotExampleSelectorBase<T>
{
    private readonly Func<string, double[]> _embeddingFunction;
    private readonly Dictionary<FewShotExample, double[]> _exampleEmbeddings;
    private readonly double _diversityThreshold;

    /// <summary>
    /// Initializes a new instance of the DiversityExampleSelector class.
    /// </summary>
    /// <param name="embeddingFunction">Function to convert text to embedding vectors.</param>
    /// <param name="diversityThreshold">Minimum dissimilarity required between selected examples (0.0 to 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The diversity threshold controls how different examples must be.
    ///
    /// - 0.0: No diversity requirement (like random selection)
    /// - 0.3: Examples must be somewhat different
    /// - 0.7: Examples must be very different
    /// - 1.0: Examples must be completely different (hard to achieve)
    ///
    /// Higher values = more diversity but may be harder to satisfy.
    /// </para>
    /// </remarks>
    public DiversityExampleSelector(Func<string, double[]> embeddingFunction, double diversityThreshold = 0.3)
    {
        _embeddingFunction = embeddingFunction ?? throw new ArgumentNullException(nameof(embeddingFunction));
        _diversityThreshold = Math.Max(0, Math.Min(1, diversityThreshold));
        _exampleEmbeddings = new Dictionary<FewShotExample, double[]>();
    }

    /// <summary>
    /// Gets the diversity threshold used for selection.
    /// </summary>
    public double DiversityThreshold => _diversityThreshold;

    /// <summary>
    /// Called when an example is added. Pre-computes the embedding.
    /// </summary>
    protected override void OnExampleAdded(FewShotExample example)
    {
        _exampleEmbeddings[example] = _embeddingFunction(example.Input);
    }

    /// <summary>
    /// Called when an example is removed. Removes the cached embedding.
    /// </summary>
    protected override void OnExampleRemoved(FewShotExample example)
    {
        _exampleEmbeddings.Remove(example);
    }

    /// <summary>
    /// Selects diverse examples using a greedy algorithm.
    /// </summary>
    protected override IReadOnlyList<FewShotExample> SelectExamplesCore(string query, int count)
    {
        var selected = new List<FewShotExample>();
        var remaining = new List<FewShotExample>(Examples);

        // Start with a random example
        if (remaining.Count == 0)
        {
            return selected.AsReadOnly();
        }

        var random = new Random();
        var firstIndex = random.Next(remaining.Count);
        selected.Add(remaining[firstIndex]);
        remaining.RemoveAt(firstIndex);

        // Greedily select the most different examples
        while (selected.Count < count && remaining.Count > 0)
        {
            FewShotExample? bestCandidate = null;
            double bestMinDissimilarity = -1;

            foreach (var candidate in remaining)
            {
                var candidateEmbedding = _exampleEmbeddings[candidate];

                // Calculate minimum dissimilarity to all selected examples
                double minDissimilarity = double.MaxValue;
                foreach (var selectedExample in selected)
                {
                    var selectedEmbedding = _exampleEmbeddings[selectedExample];
                    var similarity = CosineSimilarity(candidateEmbedding, selectedEmbedding);
                    var dissimilarity = 1 - similarity;
                    minDissimilarity = Math.Min(minDissimilarity, dissimilarity);
                }

                // Select the candidate with the highest minimum dissimilarity
                if (minDissimilarity > bestMinDissimilarity)
                {
                    bestMinDissimilarity = minDissimilarity;
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate is not null && bestMinDissimilarity >= _diversityThreshold)
            {
                selected.Add(bestCandidate);
                remaining.Remove(bestCandidate);
            }
            else if (bestCandidate is not null)
            {
                // If we can't meet the threshold, still add the best candidate
                selected.Add(bestCandidate);
                remaining.Remove(bestCandidate);
            }
            else
            {
                break;
            }
        }

        return selected.AsReadOnly();
    }

    /// <summary>
    /// Calculates cosine similarity between two vectors.
    /// </summary>
    private static double CosineSimilarity(double[] a, double[] b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length.");
        }

        double dotProduct = 0;
        double magnitudeA = 0;
        double magnitudeB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        double magnitude = Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB);
        return magnitude > 0 ? dotProduct / magnitude : 0;
    }
}
