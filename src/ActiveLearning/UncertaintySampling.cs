using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements uncertainty sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Uncertainty sampling is one of the simplest and most popular
/// active learning strategies. It selects samples where the model is least confident about
/// its predictions. The intuition is that uncertain samples are near the decision boundary
/// and provide the most information for learning.</para>
///
/// <para><b>Uncertainty measures:</b></para>
/// <list type="bullet">
/// <item><description><b>Least Confidence:</b> 1 - max(probabilities). Selects samples where
/// the top prediction has low probability.</description></item>
/// <item><description><b>Margin Sampling:</b> P(1st) - P(2nd). Selects samples where the gap
/// between the top two predictions is small.</description></item>
/// <item><description><b>Entropy:</b> -Σ p*log(p). Selects samples where the probability
/// distribution is spread out (high entropy = high uncertainty).</description></item>
/// </list>
///
/// <para><b>Reference:</b> Lewis and Gale, "A Sequential Algorithm for Training Text Classifiers" (1994).</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Sequential Algorithm for Training Text Classifiers", "https://doi.org/10.1007/978-1-4471-2099-5_1", Year = 1994, Authors = "David D. Lewis, William A. Gale")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class UncertaintySampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly UncertaintyMeasure _measure;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines the uncertainty measure to use.
    /// </summary>
    public enum UncertaintyMeasure
    {
        /// <summary>1 - max(probabilities)</summary>
        LeastConfidence,
        /// <summary>P(1st) - P(2nd)</summary>
        MarginSampling,
        /// <summary>-Σ p*log(p)</summary>
        Entropy
    }

    /// <summary>
    /// Initializes a new instance of the UncertaintySampling class.
    /// </summary>
    /// <param name="measure">The uncertainty measure to use (default: Entropy).</param>
    public UncertaintySampling(UncertaintyMeasure measure = UncertaintyMeasure.Entropy)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _measure = measure;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"UncertaintySampling-{_measure}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var scores = ComputeInformativenessScores(model, unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        if (_useBatchDiversity)
        {
            return SelectWithDiversity(scores, unlabeledPool, batchSize);
        }
        else
        {
            return SelectTopScoring(scores, batchSize);
        }
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var predictions = model.Predict(unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        var numClasses = predictions.Length / numSamples;
        var probs = ActiveLearningEngineOps.SoftmaxRows(predictions, numSamples, numClasses);
        var scores = ComputeUncertainty(probs).ToVector();

        UpdateStatistics(scores);
        return scores;
    }

    /// <inheritdoc />
    public Dictionary<string, T> GetSelectionStatistics()
    {
        return new Dictionary<string, T>
        {
            ["MinScore"] = _lastMinScore,
            ["MaxScore"] = _lastMaxScore,
            ["MeanScore"] = _lastMeanScore
        };
    }

    /// <summary>
    /// Computes the uncertainty of every sample based on the configured measure.
    /// </summary>
    /// <param name="probabilities">A [numSamples, numClasses] tensor of class probabilities.</param>
    private Tensor<T> ComputeUncertainty(Tensor<T> probabilities)
    {
        var engine = AiDotNetEngine.Current;
        switch (_measure)
        {
            case UncertaintyMeasure.LeastConfidence:
                // 1 - max(probabilities)
                return engine.ScalarMinusTensor(_numOps.One, ActiveLearningEngineOps.MaxRows(probabilities));
            case UncertaintyMeasure.MarginSampling:
                {
                    // 1 - (P(1st) - P(2nd))
                    var (first, second) = ActiveLearningEngineOps.TopTwoRows(probabilities);
                    return engine.ScalarMinusTensor(_numOps.One, engine.TensorSubtract(first, second));
                }
            default:
                // Entropy: -Σ p*log(p)
                return ActiveLearningEngineOps.EntropyRows(probabilities, _numOps.FromDouble(1e-10));
        }
    }

    /// <summary>
    /// Selects top-scoring samples without diversity consideration.
    /// </summary>
    private static int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Selects samples considering both uncertainty and diversity.
    /// </summary>
    private static int[] SelectWithDiversity(Vector<T> scores, Tensor<T> pool, int batchSize)
    {
        return ActiveLearningEngineOps.SelectWithDiversity(scores, pool, batchSize);
    }

    /// <summary>
    /// Updates selection statistics.
    /// </summary>
    private void UpdateStatistics(Vector<T> scores)
    {
        if (scores.Length == 0) return;

        (_lastMinScore, _lastMaxScore, _lastMeanScore) = ActiveLearningEngineOps.Statistics(scores);
    }
}
