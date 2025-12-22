using AiDotNet.ActiveLearning.Config;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Strategies.Uncertainty;

/// <summary>
/// Uncertainty sampling strategy for active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Uncertainty sampling is the simplest and most popular
/// active learning strategy. It selects samples where the model is most uncertain
/// about its prediction.</para>
///
/// <para><b>Uncertainty Measures:</b></para>
/// <list type="bullet">
/// <item><description><b>Entropy:</b> H(p) = -Σ p_i * log(p_i) - uncertainty across all classes</description></item>
/// <item><description><b>Least Confidence:</b> 1 - max(p) - uncertainty in top prediction</description></item>
/// <item><description><b>Margin:</b> p_1 - p_2 - difference between top two predictions</description></item>
/// </list>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Good baseline for most classification problems</description></item>
/// <item><description>Fast to compute (single forward pass)</description></item>
/// <item><description>Works well when model uncertainty correlates with true difficulty</description></item>
/// </list>
///
/// <para><b>Reference:</b> Lewis and Gale "A Sequential Algorithm for Training Text Classifiers" (1994)</para>
/// </remarks>
public class UncertaintySamplingStrategy<T, TInput, TOutput> : IUncertaintyStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly UncertaintyMeasure _uncertaintyMeasure;
    private readonly ActiveLearnerConfig<T>? _config;

    /// <inheritdoc/>
    public string Name => $"UncertaintySampling ({_uncertaintyMeasure})";

    /// <inheritdoc/>
    public string Description => _uncertaintyMeasure switch
    {
        UncertaintyMeasure.Entropy => "Selects samples with highest prediction entropy",
        UncertaintyMeasure.LeastConfidence => "Selects samples with lowest maximum prediction probability",
        UncertaintyMeasure.Margin => "Selects samples with smallest margin between top two predictions",
        UncertaintyMeasure.PredictiveVariance => "Selects samples with highest predictive variance",
        _ => "Selects samples where model is most uncertain"
    };

    /// <summary>
    /// Initializes a new uncertainty sampling strategy with default entropy measure.
    /// </summary>
    public UncertaintySamplingStrategy()
        : this(UncertaintyMeasure.Entropy, null)
    {
    }

    /// <summary>
    /// Initializes a new uncertainty sampling strategy with specified measure.
    /// </summary>
    /// <param name="uncertaintyMeasure">The uncertainty measure to use.</param>
    /// <param name="config">Optional configuration.</param>
    public UncertaintySamplingStrategy(UncertaintyMeasure uncertaintyMeasure, ActiveLearnerConfig<T>? config = null)
    {
        _uncertaintyMeasure = uncertaintyMeasure;
        _config = config;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeScores(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool)
    {
        var scores = new T[unlabeledPool.Count];

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            scores[i] = ComputeUncertainty(model, input);
        }

        return new Vector<T>(scores);
    }

    /// <inheritdoc/>
    public int[] SelectSamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        var scores = ComputeScores(model, unlabeledPool);
        var batchSizeToUse = Math.Min(batchSize, unlabeledPool.Count);

        // Sort by score descending and take top batchSize
        var indexedScores = scores
            .Select((score, index) => (Index: index, Score: score))
            .OrderByDescending(x => NumOps.ToDouble(x.Score))
            .Take(batchSizeToUse)
            .Select(x => x.Index)
            .ToArray();

        return indexedScores;
    }

    /// <inheritdoc/>
    public T ComputeUncertainty(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        var probabilities = GetPredictionProbabilities(model, input);
        return ComputeUncertaintyFromProbabilities(probabilities);
    }

    /// <inheritdoc/>
    public Vector<T> GetPredictionProbabilities(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        // Get raw predictions from model
        var prediction = model.Predict(input);

        // Convert prediction to probability distribution
        // This assumes the model can provide probabilities
        if (model is IProbabilisticModel<T, TInput, TOutput> probModel)
        {
            return probModel.PredictProbabilities(input);
        }

        // Fallback: use prediction as-is if it's already a vector
        if (prediction is Vector<T> vectorPrediction)
        {
            return Softmax(vectorPrediction);
        }

        // For single-output models, create binary probability
        var outputValue = ConvertToNumeric(prediction);
        var p = Sigmoid(outputValue);
        var oneMinusP = NumOps.Subtract(NumOps.One, p);
        return new Vector<T>(new[] { oneMinusP, p });
    }

    /// <inheritdoc/>
    public void UpdateState(int[] newlyLabeledIndices, TOutput[] labels)
    {
        // Uncertainty sampling is stateless - no update needed
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // Uncertainty sampling is stateless - no reset needed
    }

    #region Private Methods

    private T ComputeUncertaintyFromProbabilities(Vector<T> probabilities)
    {
        return _uncertaintyMeasure switch
        {
            UncertaintyMeasure.Entropy => ComputeEntropy(probabilities),
            UncertaintyMeasure.LeastConfidence => ComputeLeastConfidence(probabilities),
            UncertaintyMeasure.Margin => ComputeMargin(probabilities),
            UncertaintyMeasure.PredictiveVariance => ComputeVariance(probabilities),
            _ => ComputeEntropy(probabilities)
        };
    }

    private T ComputeEntropy(Vector<T> probabilities)
    {
        // H(p) = -Σ p_i * log(p_i)
        T entropy = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < probabilities.Length; i++)
        {
            var p = probabilities[i];
            if (NumOps.Compare(p, NumOps.Zero) > 0)
            {
                var pClipped = NumOps.Add(p, epsilon);
                var logP = NumOps.Log(pClipped);
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(p, logP));
            }
        }

        return entropy;
    }

    private T ComputeLeastConfidence(Vector<T> probabilities)
    {
        // 1 - max(p)
        T maxProb = NumOps.Zero;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (NumOps.Compare(probabilities[i], maxProb) > 0)
            {
                maxProb = probabilities[i];
            }
        }
        return NumOps.Subtract(NumOps.One, maxProb);
    }

    private T ComputeMargin(Vector<T> probabilities)
    {
        // 1 - (p_1 - p_2) where p_1 >= p_2 are top two probabilities
        if (probabilities.Length < 2)
        {
            return NumOps.One; // Maximum uncertainty for single-class
        }

        // Find top two probabilities
        T first = NumOps.MinValue;
        T second = NumOps.MinValue;

        for (int i = 0; i < probabilities.Length; i++)
        {
            if (NumOps.Compare(probabilities[i], first) > 0)
            {
                second = first;
                first = probabilities[i];
            }
            else if (NumOps.Compare(probabilities[i], second) > 0)
            {
                second = probabilities[i];
            }
        }

        // Return 1 - margin (so higher uncertainty = higher score)
        var margin = NumOps.Subtract(first, second);
        return NumOps.Subtract(NumOps.One, margin);
    }

    private T ComputeVariance(Vector<T> probabilities)
    {
        // Variance of the probability distribution
        T mean = NumOps.Divide(NumOps.One, NumOps.FromDouble(probabilities.Length));
        T variance = NumOps.Zero;

        for (int i = 0; i < probabilities.Length; i++)
        {
            var diff = NumOps.Subtract(probabilities[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(variance, NumOps.FromDouble(probabilities.Length));
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        // Find max for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.Compare(logits[i], maxLogit) > 0)
            {
                maxLogit = logits[i];
            }
        }

        // Compute exp(x - max) and sum
        var expValues = new T[logits.Length];
        T sum = NumOps.Zero;

        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.Subtract(logits[i], maxLogit);
            expValues[i] = NumOps.Exp(shifted);
            sum = NumOps.Add(sum, expValues[i]);
        }

        // Normalize
        for (int i = 0; i < expValues.Length; i++)
        {
            expValues[i] = NumOps.Divide(expValues[i], sum);
        }

        return new Vector<T>(expValues);
    }

    private T Sigmoid(T x)
    {
        // 1 / (1 + exp(-x))
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        var denominator = NumOps.Add(NumOps.One, expNegX);
        return NumOps.Divide(NumOps.One, denominator);
    }

    private T ConvertToNumeric(TOutput output)
    {
        // Try to convert output to numeric type
        if (output is T typedValue)
        {
            return typedValue;
        }

        if (output is double doubleValue)
        {
            return NumOps.FromDouble(doubleValue);
        }

        if (output is float floatValue)
        {
            return NumOps.FromDouble(floatValue);
        }

        if (output is int intValue)
        {
            return NumOps.FromDouble(intValue);
        }

        // Default fallback
        return NumOps.Zero;
    }

    #endregion
}

/// <summary>
/// Interface for models that can provide probability distributions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IProbabilisticModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Predicts class probabilities for an input.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <returns>Probability distribution over classes.</returns>
    Vector<T> PredictProbabilities(TInput input);
}
