using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Reasoning.Aggregation;

/// <summary>
/// Aggregates answers using majority voting - the most common answer wins.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Majority voting is like a democratic election - count all the answers
/// and whichever answer appears most often wins.
///
/// **Example:**
/// Given 10 answers: ["36", "36", "35", "36", "36", "36", "37", "36", "36", "36"]
/// - "36" appears 8 times
/// - "35" appears 1 time
/// - "37" appears 1 time
/// - Winner: "36" (majority)
///
/// **Why it works:**
/// - Random errors are unlikely to repeat
/// - Correct reasoning often leads to the same answer
/// - Reduces impact of outliers or mistakes
///
/// **Used in:**
/// - Self-Consistency with CoT (Wang et al., 2022)
/// - Ensemble methods
/// - Multi-path reasoning
///
/// This is one of the simplest but most effective techniques for improving reasoning accuracy.
/// </para>
/// </remarks>
public class MajorityVotingAggregator<T> : IAnswerAggregator<T>
{
    /// <inheritdoc/>
    public string MethodName => "Majority Voting";

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method counts how many times each answer appears,
    /// then returns the one that appeared most frequently. Confidence scores are not used
    /// in pure majority voting - every vote counts equally.
    /// </para>
    /// </remarks>
    public string Aggregate(List<string> answers, Vector<T> confidenceScores)
    {
        if (answers == null || answers.Count == 0)
            throw new ArgumentException("Answers list cannot be null or empty", nameof(answers));

        // Count occurrences of each answer
        var answerCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var answer in answers)
        {
            if (string.IsNullOrWhiteSpace(answer))
                continue;

            string normalized = answer.Trim();
            if (answerCounts.ContainsKey(normalized))
            {
                answerCounts[normalized]++;
            }
            else
            {
                answerCounts[normalized] = 1;
            }
        }

        if (answerCounts.Count == 0)
            throw new InvalidOperationException("No valid answers to aggregate");

        // Find the answer with the most votes
        var winner = answerCounts.OrderByDescending(kvp => kvp.Value).First();

        return winner.Key;
    }
}
