using AiDotNet.ActiveLearning.Config;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Strategies.Bayesian;

/// <summary>
/// Bayesian Active Learning by Disagreement (BALD) strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BALD uses Bayesian principles to select samples that would
/// provide the most information about the model's parameters. It measures the mutual
/// information between predictions and model parameters.</para>
///
/// <para><b>How BALD Works:</b></para>
/// <list type="number">
/// <item><description>Use Monte Carlo Dropout to sample from the posterior</description></item>
/// <item><description>Compute predictive entropy H[y|x,D]</description></item>
/// <item><description>Compute expected posterior entropy E[H[y|x,w]]</description></item>
/// <item><description>Select samples with highest mutual information: I[y,w|x,D] = H[y|x,D] - E[H[y|x,w]]</description></item>
/// </list>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Model supports dropout or other uncertainty estimation</description></item>
/// <item><description>Want to distinguish between model uncertainty and data noise</description></item>
/// <item><description>Deep learning models with dropout layers</description></item>
/// </list>
///
/// <para><b>Reference:</b> Gal et al. "Deep Bayesian Active Learning with Image Data" (ICML 2017)</para>
/// </remarks>
public class BALDStrategy<T, TInput, TOutput> : IBayesianStrategy<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _mcSamples;
    private readonly T _dropoutRate;
    private readonly ActiveLearnerConfig<T>? _config;

    /// <inheritdoc/>
    public string Name => "BALD (Bayesian Active Learning by Disagreement)";

    /// <inheritdoc/>
    public string Description =>
        "Selects samples with highest mutual information between predictions and model parameters using MC Dropout";

    /// <inheritdoc/>
    public int MonteCarloSamples => _mcSamples;

    /// <summary>
    /// Initializes a new BALD strategy with default parameters.
    /// </summary>
    public BALDStrategy()
        : this(20, 0.5, null)
    {
    }

    /// <summary>
    /// Initializes a new BALD strategy with specified parameters.
    /// </summary>
    /// <param name="mcSamples">Number of Monte Carlo dropout samples.</param>
    /// <param name="dropoutRate">Dropout rate for uncertainty estimation.</param>
    /// <param name="config">Optional configuration.</param>
    public BALDStrategy(int mcSamples, double dropoutRate, ActiveLearnerConfig<T>? config = null)
    {
        _mcSamples = mcSamples > 0 ? mcSamples : 20;
        _dropoutRate = NumOps.FromDouble(dropoutRate > 0 && dropoutRate < 1 ? dropoutRate : 0.5);
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
            scores[i] = ComputeMutualInformation(model, input);
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

        var indexedScores = scores
            .Select((score, index) => (Index: index, Score: score))
            .OrderByDescending(x => NumOps.ToDouble(x.Score))
            .Take(batchSizeToUse)
            .Select(x => x.Index)
            .ToArray();

        return indexedScores;
    }

    /// <inheritdoc/>
    public T ComputeMutualInformation(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        // I[y,w|x,D] = H[y|x,D] - E_w[H[y|x,w]]
        // = Predictive Entropy - Expected Entropy under posterior

        // Collect MC dropout samples
        var mcPredictions = CollectMCPredictions(model, input);

        // Compute predictive entropy (entropy of averaged predictions)
        var avgPrediction = ComputeAveragePrediction(mcPredictions);
        var predictiveEntropy = ComputeEntropy(avgPrediction);

        // Compute expected entropy (average entropy of individual predictions)
        var expectedEntropy = ComputeExpectedEntropy(mcPredictions);

        // Mutual information = predictive entropy - expected entropy
        return NumOps.Subtract(predictiveEntropy, expectedEntropy);
    }

    /// <inheritdoc/>
    public T ComputePredictiveEntropy(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        var mcPredictions = CollectMCPredictions(model, input);
        var avgPrediction = ComputeAveragePrediction(mcPredictions);
        return ComputeEntropy(avgPrediction);
    }

    /// <inheritdoc/>
    public void UpdateState(int[] newlyLabeledIndices, TOutput[] labels)
    {
        // BALD is stateless - no update needed
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // BALD is stateless - no reset needed
    }

    #region Private Methods

    private List<Vector<T>> CollectMCPredictions(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        var predictions = new List<Vector<T>>();

        // If model supports dropout/stochastic forward passes
        if (model is IDropoutModel<T, TInput, TOutput> dropoutModel)
        {
            for (int i = 0; i < _mcSamples; i++)
            {
                var prediction = dropoutModel.PredictWithDropout(input, _dropoutRate);
                var probabilities = ConvertToProbabilities(prediction);
                predictions.Add(probabilities);
            }
        }
        else
        {
            // Fallback: simulate uncertainty with noise
            for (int i = 0; i < _mcSamples; i++)
            {
                var prediction = model.Predict(input);
                var probabilities = ConvertToProbabilities(prediction);
                var noisyProbabilities = AddNoise(probabilities);
                predictions.Add(noisyProbabilities);
            }
        }

        return predictions;
    }

    private Vector<T> ComputeAveragePrediction(List<Vector<T>> predictions)
    {
        if (predictions.Count == 0)
        {
            return new Vector<T>(0);
        }

        var length = predictions[0].Length;
        var avg = new T[length];

        for (int i = 0; i < length; i++)
        {
            avg[i] = NumOps.Zero;
        }

        foreach (var pred in predictions)
        {
            for (int i = 0; i < length; i++)
            {
                avg[i] = NumOps.Add(avg[i], pred[i]);
            }
        }

        var count = NumOps.FromDouble(predictions.Count);
        for (int i = 0; i < length; i++)
        {
            avg[i] = NumOps.Divide(avg[i], count);
        }

        return new Vector<T>(avg);
    }

    private T ComputeExpectedEntropy(List<Vector<T>> predictions)
    {
        if (predictions.Count == 0)
        {
            return NumOps.Zero;
        }

        T totalEntropy = NumOps.Zero;
        foreach (var pred in predictions)
        {
            totalEntropy = NumOps.Add(totalEntropy, ComputeEntropy(pred));
        }

        return NumOps.Divide(totalEntropy, NumOps.FromDouble(predictions.Count));
    }

    private T ComputeEntropy(Vector<T> probabilities)
    {
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

    private Vector<T> ConvertToProbabilities(TOutput prediction)
    {
        if (prediction is Vector<T> vectorPred)
        {
            return Softmax(vectorPred);
        }

        // For scalar predictions, create binary probability
        var value = ConvertToNumeric(prediction);
        var p = Sigmoid(value);
        var oneMinusP = NumOps.Subtract(NumOps.One, p);
        return new Vector<T>(new[] { oneMinusP, p });
    }

    private Vector<T> AddNoise(Vector<T> probabilities)
    {
        // Add small Gaussian noise to simulate uncertainty
        var noisy = new T[probabilities.Length];
        T sum = NumOps.Zero;

        for (int i = 0; i < probabilities.Length; i++)
        {
            var noise = NumOps.FromDouble((ThreadRandom.NextDouble() - 0.5) * 0.1);
            noisy[i] = NumOps.Add(probabilities[i], noise);
            // Ensure non-negative
            if (NumOps.Compare(noisy[i], NumOps.Zero) < 0)
            {
                noisy[i] = NumOps.FromDouble(1e-10);
            }
            sum = NumOps.Add(sum, noisy[i]);
        }

        // Renormalize
        for (int i = 0; i < noisy.Length; i++)
        {
            noisy[i] = NumOps.Divide(noisy[i], sum);
        }

        return new Vector<T>(noisy);
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.Compare(logits[i], maxLogit) > 0)
            {
                maxLogit = logits[i];
            }
        }

        var expValues = new T[logits.Length];
        T sum = NumOps.Zero;

        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.Subtract(logits[i], maxLogit);
            expValues[i] = NumOps.Exp(shifted);
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < expValues.Length; i++)
        {
            expValues[i] = NumOps.Divide(expValues[i], sum);
        }

        return new Vector<T>(expValues);
    }

    private T Sigmoid(T x)
    {
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        var denominator = NumOps.Add(NumOps.One, expNegX);
        return NumOps.Divide(NumOps.One, denominator);
    }

    private T ConvertToNumeric(TOutput output)
    {
        if (output is T typedValue) return typedValue;
        if (output is double doubleValue) return NumOps.FromDouble(doubleValue);
        if (output is float floatValue) return NumOps.FromDouble(floatValue);
        if (output is int intValue) return NumOps.FromDouble(intValue);
        return NumOps.Zero;
    }

    #endregion
}

/// <summary>
/// Interface for models that support dropout inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IDropoutModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Makes a prediction with dropout enabled for uncertainty estimation.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <param name="dropoutRate">The dropout rate to use.</param>
    /// <returns>The prediction with dropout applied.</returns>
    TOutput PredictWithDropout(TInput input, T dropoutRate);

    /// <summary>
    /// Enables or disables dropout for inference.
    /// </summary>
    /// <param name="enabled">Whether to enable dropout.</param>
    void SetDropoutEnabled(bool enabled);
}
