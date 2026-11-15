using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements margin sampling for active learning sample selection.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Margin sampling selects samples where the difference between
/// the top two predicted class probabilities is smallest. A small margin indicates the model
/// is uncertain between two classes, suggesting the sample is near the decision boundary.</para>
///
/// <para><b>Formula:</b> margin = P(y₁|x) - P(y₂|x), where y₁ and y₂ are the most likely classes.</para>
/// <para><b>Selection:</b> Select samples with smallest margins (transformed as 1 - margin for
/// consistency with other strategies where higher = more informative).</para>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Simple and intuitive interpretation.</description></item>
/// <item><description>Focuses on decision boundary uncertainty.</description></item>
/// <item><description>Works well for multi-class classification.</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n × c) where n=pool size, c=number of classes.</para>
/// <para><b>Reference:</b> Settles, B. (2012). "Active Learning." Morgan &amp; Claypool Publishers.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Active Learning", "https://doi.org/10.1007/978-3-031-01560-1", Year = 2012, Authors = "Burr Settles")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class MarginSampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the MarginSampling class.
    /// </summary>
    public MarginSampling()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => "MarginSampling";

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
        var engine = AiDotNetEngine.Current;
        var probs = ActiveLearningEngineOps.SoftmaxRows(predictions, numSamples, numClasses);
        var (first, second) = ActiveLearningEngineOps.TopTwoRows(probs);
        var margin = engine.TensorSubtract(first, second);
        var scores = engine.ScalarMinusTensor(_numOps.One, margin).ToVector();

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
    /// Selects top-scoring samples without diversity consideration.
    /// </summary>
    private static int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Selects samples considering both margin score and diversity.
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
