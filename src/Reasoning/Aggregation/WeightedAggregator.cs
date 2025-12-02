using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Reasoning.Aggregation;

/// <summary>
/// Aggregates answers using weighted voting based on confidence scores.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Weighted voting is like voting where some votes count more than others
/// based on confidence. Answers with higher confidence scores get more weight.
///
/// **Example:**
/// Answers and confidence scores:
/// - "36" with confidence 0.9
/// - "36" with confidence 0.8
/// - "35" with confidence 0.5
/// - "36" with confidence 0.95
///
/// Total weight for "36": 0.9 + 0.8 + 0.95 = 2.65
/// Total weight for "35": 0.5
/// Winner: "36" (highest total weight)
///
/// **Difference from Majority Voting:**
/// - Majority: Each vote counts equally (1, 1, 1...)
/// - Weighted: Votes count based on confidence (0.9, 0.8, 0.5...)
///
/// **When to use:**
/// - When confidence scores are reliable
/// - When you want high-quality answers to matter more
/// - When combining results from different reasoning paths
/// </para>
/// </remarks>
internal class WeightedAggregator<T> : IAnswerAggregator<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="WeightedAggregator{T}"/> class.
    /// </summary>
    public WeightedAggregator()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string MethodName => "Weighted Voting";

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method counts answers but weights each one by its
    /// confidence score. An answer with confidence 0.9 counts more than one with confidence 0.3.
    /// </para>
    /// </remarks>
    public string Aggregate(List<string> answers, Vector<T> confidenceScores)
    {
        if (answers == null || answers.Count == 0)
            throw new ArgumentException("Answers list cannot be null or empty", nameof(answers));

        if (confidenceScores == null || confidenceScores.Length != answers.Count)
            throw new ArgumentException("Confidence scores must match answers length", nameof(confidenceScores));

        // Accumulate weighted scores for each answer
        var answerWeights = new Dictionary<string, T>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < answers.Count; i++)
        {
            var answer = answers[i];
            if (string.IsNullOrWhiteSpace(answer))
                continue;

            string normalized = answer.Trim();
            T score = confidenceScores[i];

            if (answerWeights.ContainsKey(normalized))
            {
                answerWeights[normalized] = _numOps.Add(answerWeights[normalized], score);
            }
            else
            {
                answerWeights[normalized] = score;
            }
        }

        if (answerWeights.Count == 0)
            throw new InvalidOperationException("No valid answers to aggregate");

        // Find the answer with the highest total weight
        var winner = answerWeights.OrderByDescending(kvp => _numOps.ToDouble(kvp.Value)).First();

        return winner.Key;
    }
}
