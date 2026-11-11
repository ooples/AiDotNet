using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements least confidence sampling for active learning sample selection.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Least confidence sampling selects samples where the model's
/// top prediction has the lowest probability. This is the simplest uncertainty measure,
/// focusing on how confident the model is in its best guess.</para>
///
/// <para><b>Formula:</b> LC(x) = 1 - max P(y|x)</para>
///
/// <para><b>Interpretation:</b></para>
/// <list type="bullet">
/// <item><description>LC = 0: Model is 100% confident.</description></item>
/// <item><description>LC = (k-1)/k: Maximum uncertainty for k classes (uniform distribution).</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Simplest uncertainty measure.</description></item>
/// <item><description>Very efficient to compute.</description></item>
/// <item><description>Works for any classifier that outputs probabilities.</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n × c) where n=pool size, c=number of classes.</para>
/// <para><b>Reference:</b> Lewis, D.D. &amp; Catlett, J. (1994). "Heterogeneous uncertainty
/// sampling for supervised learning."</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Heterogeneous Uncertainty Sampling for Supervised Learning", "https://doi.org/10.1016/B978-1-55860-335-6.50026-X", Year = 1994, Authors = "David D. Lewis, Jason Catlett")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class LeastConfidenceSampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the LeastConfidenceSampling class.
    /// </summary>
    public LeastConfidenceSampling()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => "LeastConfidenceSampling";

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
        var scores = new Vector<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            var probs = ExtractProbabilities(predictions, i, numClasses);
            scores[i] = ComputeLeastConfidence(probs);
        }

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
    /// Computes least confidence score: LC = 1 - max(p).
    /// </summary>
    private T ComputeLeastConfidence(Vector<T> probabilities)
    {
        var maxProb = _numOps.Zero;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (_numOps.GreaterThan(probabilities[i], maxProb))
            {
                maxProb = probabilities[i];
            }
        }
        return _numOps.Subtract(_numOps.One, maxProb);
    }

    /// <summary>
    /// Extracts probabilities for a single sample from batch predictions.
    /// </summary>
    private Vector<T> ExtractProbabilities(Tensor<T> predictions, int sampleIndex, int numClasses)
    {
        var probs = new Vector<T>(numClasses);
        var startIdx = sampleIndex * numClasses;

        // Apply softmax to get probabilities
        var maxLogit = _numOps.MinValue;
        for (int c = 0; c < numClasses; c++)
        {
            if (_numOps.GreaterThan(predictions[startIdx + c], maxLogit))
            {
                maxLogit = predictions[startIdx + c];
            }
        }

        var expSum = _numOps.Zero;
        for (int c = 0; c < numClasses; c++)
        {
            var shifted = _numOps.Subtract(predictions[startIdx + c], maxLogit);
            var expVal = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
            probs[c] = expVal;
            expSum = _numOps.Add(expSum, expVal);
        }

        for (int c = 0; c < numClasses; c++)
        {
            probs[c] = _numOps.Divide(probs[c], expSum);
        }

        return probs;
    }

    /// <summary>
    /// Selects top-scoring samples without diversity consideration.
    /// </summary>
    private static int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Selects samples considering both least confidence and diversity.
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
