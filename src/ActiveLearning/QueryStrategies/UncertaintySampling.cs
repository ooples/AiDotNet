using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning.QueryStrategies;

/// <summary>
/// Uncertainty sampling query strategy for active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Uncertainty sampling selects examples where the model
/// is most uncertain about its predictions. The intuition is that these examples
/// contain the most information for the model to learn from.</para>
///
/// <para><b>Three main variants:</b>
/// 1. <b>Least Confidence:</b> Select examples with lowest confidence in predicted class
///    - Score = 1 - P(y_max | x)
///    - Simple and effective for multi-class classification
///
/// 2. <b>Margin Sampling:</b> Select examples with smallest margin between top 2 classes
///    - Score = P(y_1 | x) - P(y_2 | x)  (smaller is more uncertain)
///    - Better for binary and multi-class when close calls matter
///
/// 3. <b>Entropy:</b> Select examples with highest prediction entropy
///    - Score = -Σ P(y | x) * log P(y | x)
///    - Considers full distribution, not just top predictions
/// </para>
///
/// <para><b>When to use which:</b>
/// - Least Confidence: Simple, works well in practice, fastest
/// - Margin: Better when distinguishing between top candidates matters
/// - Entropy: Most principled, considers all classes, slightly slower
/// </para>
/// </remarks>
public class UncertaintySampling<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Uncertainty measure to use.
    /// </summary>
    public enum UncertaintyMeasure
    {
        /// <summary>
        /// Least confidence: 1 - max(P(y|x))
        /// </summary>
        LeastConfidence,

        /// <summary>
        /// Margin: difference between top two probabilities
        /// </summary>
        Margin,

        /// <summary>
        /// Entropy: -Σ P(y|x) * log(P(y|x))
        /// </summary>
        Entropy
    }

    private readonly UncertaintyMeasure _measure;

    /// <summary>
    /// Initializes a new uncertainty sampling strategy.
    /// </summary>
    /// <param name="measure">The uncertainty measure to use.</param>
    public UncertaintySampling(UncertaintyMeasure measure = UncertaintyMeasure.LeastConfidence)
    {
        _measure = measure;
    }

    /// <inheritdoc/>
    public string Name => $"UncertaintySampling-{_measure}";

    /// <inheritdoc/>
    public T[] ScoreExamples(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> unlabeledData)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (unlabeledData == null)
            throw new ArgumentNullException(nameof(unlabeledData));

        // In a full implementation, iterate through unlabeledData
        // For now, return placeholder scores

        int numExamples = 100; // Placeholder - would get from dataset
        var scores = new T[numExamples];

        for (int i = 0; i < numExamples; i++)
        {
            // Get predictions for example i
            // var prediction = model.Predict(unlabeledData[i]);

            // Compute uncertainty score based on measure
            scores[i] = _measure switch
            {
                UncertaintyMeasure.LeastConfidence => ComputeLeastConfidence(null!),
                UncertaintyMeasure.Margin => ComputeMargin(null!),
                UncertaintyMeasure.Entropy => ComputeEntropy(null!),
                _ => NumOps.Zero
            };
        }

        return scores;
    }

    /// <inheritdoc/>
    public int[] SelectBatch(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> unlabeledData, int k)
    {
        var scores = ScoreExamples(model, unlabeledData);

        // Select top-k examples with highest uncertainty scores
        var indexedScores = scores
            .Select((score, index) => (Score: Convert.ToDouble(score), Index: index))
            .OrderByDescending(x => x.Score)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();

        return indexedScores;
    }

    /// <summary>
    /// Computes least confidence score.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities for each class.</param>
    /// <returns>Uncertainty score (higher = more uncertain).</returns>
    /// <remarks>
    /// <para>Least confidence measures uncertainty as 1 minus the probability of the most likely class.
    /// If the model is very confident (P(y_max) close to 1), the score is low.
    /// If uncertain (P(y_max) close to 1/C for C classes), the score is high.</para>
    /// </remarks>
    private T ComputeLeastConfidence(Vector<T> probabilities)
    {
        if (probabilities == null || probabilities.Length == 0)
            return NumOps.Zero;

        // Find max probability
        T maxProb = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++)
        {
            if (Convert.ToDouble(probabilities[i]) > Convert.ToDouble(maxProb))
                maxProb = probabilities[i];
        }

        // Return 1 - max_prob
        return NumOps.Subtract(NumOps.FromDouble(1.0), maxProb);
    }

    /// <summary>
    /// Computes margin sampling score.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities for each class.</param>
    /// <returns>Uncertainty score (higher = more uncertain).</returns>
    /// <remarks>
    /// <para>Margin sampling measures the difference between the top two most likely classes.
    /// A small margin indicates the model is uncertain between two options.
    /// We return negative margin so higher scores mean more uncertainty.</para>
    /// </remarks>
    private T ComputeMargin(Vector<T> probabilities)
    {
        if (probabilities == null || probabilities.Length < 2)
            return NumOps.Zero;

        // Sort probabilities in descending order
        var sorted = probabilities.ToArray()
            .OrderByDescending(p => Convert.ToDouble(p))
            .ToArray();

        // Margin = P(y_1) - P(y_2), we return negative for "higher is more uncertain"
        var margin = NumOps.Subtract(sorted[0], sorted[1]);
        return NumOps.Negate(margin); // Negate so smaller margin = higher score
    }

    /// <summary>
    /// Computes entropy score.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities for each class.</param>
    /// <returns>Uncertainty score (higher = more uncertain).</returns>
    /// <remarks>
    /// <para>Entropy measures the uncertainty across all classes:
    /// H = -Σ P(y|x) * log P(y|x)
    /// </para>
    /// <para>Maximum entropy occurs when all classes have equal probability.
    /// Minimum entropy (0) occurs when one class has probability 1.</para>
    /// </remarks>
    private T ComputeEntropy(Vector<T> probabilities)
    {
        if (probabilities == null || probabilities.Length == 0)
            return NumOps.Zero;

        T entropy = NumOps.Zero;

        for (int i = 0; i < probabilities.Length; i++)
        {
            double prob = Convert.ToDouble(probabilities[i]);

            // Skip zero probabilities (0 * log(0) = 0 by convention)
            if (prob > 1e-10)
            {
                double logProb = Math.Log(prob);
                T term = NumOps.Multiply(
                    probabilities[i],
                    NumOps.FromDouble(logProb));
                entropy = NumOps.Subtract(entropy, term);
            }
        }

        return entropy;
    }
}
