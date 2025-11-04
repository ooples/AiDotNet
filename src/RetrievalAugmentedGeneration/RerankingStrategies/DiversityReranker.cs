using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies;

/// <summary>
/// Diversity-based reranker that prioritizes variety among retrieved documents.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Reduces redundancy by ensuring retrieved documents cover diverse aspects of the topic
/// rather than returning multiple similar documents.
/// </remarks>
public class DiversityReranker<T> : RerankerBase<T>
{
    private readonly T _diversityWeight;
    private readonly T _relevanceWeight;

    /// <summary>
    /// Initializes a new instance of the <see cref="DiversityReranker{T}"/> class.
    /// </summary>
    /// <param name="diversityWeight">Weight for diversity component (0-1).</param>
    /// <param name="relevanceWeight">Weight for relevance component (0-1).</param>
    public DiversityReranker(
        T diversityWeight,
        T relevanceWeight)
    {
        _diversityWeight = diversityWeight;
        _relevanceWeight = relevanceWeight;
    }

    /// <inheritdoc />
    public override bool ModifiesScores => true;

    /// <inheritdoc />
    protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
    {
        if (documents.Count == 0)
            return Enumerable.Empty<Document<T>>();

        var selected = new List<Document<T>>();
        var remaining = new List<Document<T>>(documents);

        // Select first document by relevance
        var first = remaining.OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : NumOps.Zero).First();
        selected.Add(first);
        remaining.Remove(first);

        // Iteratively select documents balancing relevance and diversity
        while (remaining.Count > 0)
        {
            var bestDoc = remaining[0];
            var bestScore = NumOps.FromDouble(double.NegativeInfinity);

            foreach (var doc in remaining)
            {
                // Relevance score
                var relevance = doc.HasRelevanceScore ? doc.RelevanceScore : NumOps.Zero;

                // Diversity score (minimum similarity to selected documents)
                var minSimilarity = NumOps.One;
                foreach (var selectedDoc in selected)
                {
                    var similarity = StatisticsHelper<double>.JaccardSimilarity(doc.Content, selectedDoc.Content);
                    var simT = NumOps.FromDouble(similarity);
                    if (NumOps.LessThan(simT, minSimilarity))
                    {
                        minSimilarity = simT;
                    }
                }

                // Combined score
                var score = NumOps.Add(
                    NumOps.Multiply(_relevanceWeight, relevance),
                    NumOps.Multiply(_diversityWeight, NumOps.Subtract(NumOps.One, minSimilarity))
                );

                if (NumOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    bestDoc = doc;
                }
            }

            selected.Add(bestDoc);
            remaining.Remove(bestDoc);
        }

        return selected;
    }
}
