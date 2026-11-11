using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements variation ratios for active learning sample selection.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Variation ratios measure the proportion of predicted labels
/// that are NOT in the modal (most likely) class. High variation ratio indicates high
/// uncertainty because the model is not strongly predicting any single class.</para>
///
/// <para><b>Formula:</b> VR(x) = 1 - max(P(y|x))</para>
///
/// <para><b>Interpretation:</b></para>
/// <list type="bullet">
/// <item><description>VR = 0: Model is 100% confident in one class (no variation).</description></item>
/// <item><description>VR = 0.5: For binary classification, model is maximally uncertain.</description></item>
/// <item><description>VR = (k-1)/k: For k-class classification, maximum uncertainty (uniform distribution).</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Simple and intuitive interpretation.</description></item>
/// <item><description>Works well for multi-class classification.</description></item>
/// <item><description>Computationally efficient.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Freeman, L.C. (1965). "Elementary Applied Statistics: For Students
/// in Behavioral Science."</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Elementary Applied Statistics: For Students in Behavioral Science", "https://books.google.com/books?id=0HtMAAAAMAAJ", Year = 1965, Authors = "Linton C. Freeman")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class VariationRatios<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the VariationRatios class.
    /// </summary>
    public VariationRatios()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => "VariationRatios";

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
            scores[i] = ComputeVariationRatio(probs);
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
    /// Computes variation ratio: VR = 1 - max(P(y|x)).
    /// </summary>
    private T ComputeVariationRatio(Vector<T> probabilities)
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
    /// Selects samples considering both variation ratio and diversity.
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
