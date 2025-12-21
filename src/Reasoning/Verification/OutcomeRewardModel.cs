using System.Text.RegularExpressions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Verification;

/// <summary>
/// Outcome Reward Model (ORM) that evaluates only the final answer.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> While Process Reward Models (PRM) score each step of reasoning,
/// Outcome Reward Models only care about whether you got the right final answer.
///
/// **PRM vs ORM:**
/// - **PRM**: "Step 1: Good (0.8), Step 2: Great (0.9), Step 3: Okay (0.7)" → Avg: 0.8
/// - **ORM**: "Final answer correct? Yes → 1.0, No → 0.0"
///
/// **Analogy:**
/// - **PRM**: Teacher grades your work step-by-step, partial credit possible
/// - **ORM**: Multiple choice test - right or wrong, no partial credit
///
/// **When to use ORM:**
/// - When only the final answer matters (e.g., math competition)
/// - When intermediate steps are unreliable or unavailable
/// - When you want to reward results over process
/// - Simpler and faster than PRM
///
/// **When to use PRM:**
/// - When learning the reasoning process
/// - When partial credit is important
/// - When you want to catch errors early
/// - For educational or explainable AI
///
/// **Hybrid approach (best of both):**
/// Combine PRM and ORM to reward both good reasoning AND correct answers.
///
/// **Example:**
/// ```csharp
/// var orm = new OutcomeRewardModel<double>(chatModel);
///
/// var chain = new ReasoningChain<double> {
///     FinalAnswer = "42",
///     Steps = new List<ReasoningStep<double>> { /* ... */ }
/// };
///
/// // Only cares about final answer
/// double reward = await orm.CalculateRewardAsync(chain, correctAnswer: "42");
/// Console.WriteLine($"Reward: {reward}"); // 1.0 (correct)
///
/// reward = await orm.CalculateRewardAsync(chain, correctAnswer: "43");
/// Console.WriteLine($"Reward: {reward}"); // 0.0 (incorrect)
/// ```
///
/// **Research:**
/// - "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021) - Introduced ORMs
/// - "Let's Verify Step by Step" (Lightman et al., 2023) - Compared PRM vs ORM, showed PRM is better for training
/// - "Math-Shepherd" (Wang et al., 2024) - Used hybrid PRM+ORM approach
///
/// **Key insight from research:**
/// PRMs generally outperform ORMs for training reasoning models because they provide
/// more granular feedback. However, ORMs are useful for:
/// 1. Final answer verification
/// 2. Ensemble methods (combine PRM + ORM)
/// 3. Simpler problems where process doesn't matter
/// </para>
/// </remarks>
internal class OutcomeRewardModel<T> : IRewardModel<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IChatModel<T>? _chatModel;
    private readonly INumericOperations<T> _numOps;
    private readonly bool _useSemanticSimilarity;
    private readonly double _partialCreditThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="OutcomeRewardModel{T}"/> class.
    /// </summary>
    /// <param name="chatModel">Chat model for semantic answer comparison (optional).</param>
    /// <param name="useSemanticSimilarity">Use LLM for semantic comparison vs exact match.</param>
    /// <param name="partialCreditThreshold">Threshold for partial credit (0.0-1.0, default: 0.8).</param>
    public OutcomeRewardModel(
        IChatModel<T>? chatModel = null,
        bool useSemanticSimilarity = false,
        double partialCreditThreshold = 0.8)
    {
        _chatModel = chatModel;
        _numOps = MathHelper.GetNumericOperations<T>();
        _useSemanticSimilarity = useSemanticSimilarity && chatModel != null;
        _partialCreditThreshold = partialCreditThreshold;
    }

    /// <inheritdoc/>
    public string ModelName => "Outcome Reward Model (ORM)";

    /// <inheritdoc/>
    public RewardModelType ModelType => RewardModelType.Outcome;

    /// <inheritdoc/>
    public string Description =>
        "Evaluates only the final answer, not the reasoning process. " +
        "Fast and simple. Useful for result-oriented tasks.";

    /// <summary>
    /// Calculates reward based only on the final answer.
    /// </summary>
    /// <param name="chain">The reasoning chain to evaluate.</param>
    /// <param name="correctAnswer">The correct answer (optional for unsupervised).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reward score (typically 1.0 for correct, 0.0 for incorrect).</returns>
    public async Task<T> CalculateRewardAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default)
    {
        if (chain == null)
            throw new ArgumentNullException(nameof(chain));

        // If no correct answer provided, use unsupervised reward
        if (string.IsNullOrEmpty(correctAnswer))
        {
            return await CalculateUnsupervisedRewardAsync(chain, cancellationToken);
        }

        // Compare final answer with correct answer
        string finalAnswer = chain.FinalAnswer ?? string.Empty;
        string correctAnswerNonNull = correctAnswer ?? string.Empty;

        // Try exact match first
        if (AnswersMatch(finalAnswer, correctAnswerNonNull))
        {
            return _numOps.FromDouble(1.0);
        }

        // Try numerical comparison
        if (TryNumericalComparison(finalAnswer, correctAnswerNonNull, out double numericalSimilarity))
        {
            if (numericalSimilarity >= _partialCreditThreshold)
            {
                return _numOps.FromDouble(numericalSimilarity);
            }
            return _numOps.Zero;
        }

        // Try semantic similarity (if enabled)
        if (_useSemanticSimilarity && _chatModel != null)
        {
            double semanticSimilarity = await CompareSemanticSimilarityAsync(
                finalAnswer,
                correctAnswerNonNull,
                cancellationToken
            );

            if (semanticSimilarity >= _partialCreditThreshold)
            {
                return _numOps.FromDouble(semanticSimilarity);
            }
        }

        // No match
        return _numOps.Zero;
    }

    /// <summary>
    /// Calculates reward without a correct answer (unsupervised).
    /// </summary>
    private async Task<T> CalculateUnsupervisedRewardAsync(
        ReasoningChain<T> chain,
        CancellationToken cancellationToken)
    {
        await Task.CompletedTask;  // Suppress CS1998 warning

        // Heuristic-based reward without ground truth
        double reward = 0.0;

        // 1. Completeness: Does it have a clear answer?
        if (!string.IsNullOrWhiteSpace(chain.FinalAnswer))
        {
            reward += 0.3;

            // Bonus for explicit answer indicators
            string lowerAnswer = chain.FinalAnswer.ToLowerInvariant();
            if (lowerAnswer.Contains("final answer") ||
                lowerAnswer.Contains("therefore") ||
                lowerAnswer.Contains("conclusion"))
            {
                reward += 0.1;
            }
        }

        // 2. Confidence: How confident is the model?
        if (chain.Steps.Count > 0)
        {
            var avgConfidence = chain.Steps
                .Where(s => Convert.ToDouble(s.Score) > 0)
                .Select(s => Convert.ToDouble(s.Score))
                .DefaultIfEmpty(0.0)
                .Average();

            reward += avgConfidence * 0.3;
        }

        // 3. Coherence: Are steps logically connected?
        if (chain.Steps.Count > 1)
        {
            bool hasCoherence = chain.Steps.All(s => !string.IsNullOrWhiteSpace(s.Content));
            if (hasCoherence)
            {
                reward += 0.2;
            }
        }

        // 4. Verification: Did it pass any verifiers?
        // TODO: ReasoningChain<T> doesn't have VerificationResults property yet
        // if (chain.VerificationResults?.Count > 0)
        // {
        //     double verificationScore = chain.VerificationResults
        //         .Where(v => v.Passed)
        //         .Select(v => Convert.ToDouble(v.Confidence))
        //         .DefaultIfEmpty(0.0)
        //         .Average();
        //
        //     reward += verificationScore * 0.1;
        // }

        return _numOps.FromDouble(MathHelper.Clamp(reward, 0.0, 1.0));
    }

    /// <inheritdoc/>
    public Task<T> CalculateStepRewardAsync(
        ReasoningStep<T> step,
        ReasoningContext context,
        CancellationToken cancellationToken = default)
    {
        // ORM doesn't score individual steps, only final outcome
        // Return neutral score
        return Task.FromResult(_numOps.FromDouble(0.5));
    }

    /// <inheritdoc/>
    public async Task<T> CalculateChainRewardAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default)
    {
        // Same as CalculateRewardAsync for ORM
        return await CalculateRewardAsync(chain, correctAnswer, cancellationToken);
    }

    private bool AnswersMatch(string answer1, string answer2)
    {
        // Normalize and compare
        string norm1 = NormalizeAnswer(answer1);
        string norm2 = NormalizeAnswer(answer2);

        return norm1.Equals(norm2, StringComparison.OrdinalIgnoreCase);
    }

    private string NormalizeAnswer(string answer)
    {
        if (string.IsNullOrWhiteSpace(answer))
            return string.Empty;

        // Remove common prefixes
        answer = Regex.Replace(answer, @"^(the answer is|final answer:?|therefore,?|thus,?)\s*", "", RegexOptions.IgnoreCase, RegexTimeout);

        // Remove punctuation and extra spaces
        answer = Regex.Replace(answer, @"[^\w\s\d\.]", "", RegexOptions.None, RegexTimeout);
        answer = Regex.Replace(answer, @"\s+", " ", RegexOptions.None, RegexTimeout);

        return answer.Trim();
    }

    private bool TryNumericalComparison(string answer1, string answer2, out double similarity)
    {
        similarity = 0.0;

        // Extract numbers
        var numbers1 = ExtractNumbers(answer1);
        var numbers2 = ExtractNumbers(answer2);

        if (numbers1.Count == 0 || numbers2.Count == 0)
            return false;

        // Compare primary numbers (usually the last one)
        double num1 = numbers1.Last();
        double num2 = numbers2.Last();

        // Calculate similarity based on relative difference
        double diff = Math.Abs(num1 - num2);
        double magnitude = Math.Max(Math.Abs(num1), Math.Abs(num2));

        if (magnitude < 0.0001) // Both near zero
        {
            similarity = diff < 0.0001 ? 1.0 : 0.0;
        }
        else
        {
            double relativeDiff = diff / magnitude;
            similarity = Math.Max(0.0, 1.0 - relativeDiff * 10); // Scale: 10% diff → 0.0
        }

        return true;
    }

    private List<double> ExtractNumbers(string text)
    {
        var numbers = new List<double>();

        var matches = Regex.Matches(text, @"-?\d+\.?\d*", RegexOptions.None, RegexTimeout);
        foreach (Match match in matches)
        {
            if (double.TryParse(match.Value, out double number))
            {
                numbers.Add(number);
            }
        }

        return numbers;
    }

    private async Task<double> CompareSemanticSimilarityAsync(
        string answer1,
        string answer2,
        CancellationToken cancellationToken)
    {
        if (_chatModel == null)
            return 0.0;

        // Check cancellation before making LLM call
        cancellationToken.ThrowIfCancellationRequested();

        string prompt = $@"Compare these two answers and rate their semantic similarity on a scale of 0.0 to 1.0.
1.0 means they express the same answer/meaning.
0.0 means they are completely different or contradictory.

Answer 1: {answer1}
Answer 2: {answer2}

Respond with ONLY a number between 0.0 and 1.0, nothing else.";

        try
        {
            string response = await _chatModel.GenerateResponseAsync(prompt);

            // Check cancellation after LLM call
            cancellationToken.ThrowIfCancellationRequested();

            // Extract number from response
            var match = Regex.Match(response, @"([0-1]\.?\d*)", RegexOptions.None, RegexTimeout);
            if (match.Success && double.TryParse(match.Value, out double similarity))
            {
                return MathHelper.Clamp(similarity, 0.0, 1.0);
            }
        }
        catch
        {
            // Fallback to simple comparison
        }

        return 0.0;
    }
}
