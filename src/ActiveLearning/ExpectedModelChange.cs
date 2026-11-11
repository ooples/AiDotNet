using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Expected Model Change (EMC) for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Expected Model Change selects samples that would cause the largest
/// change to the model's parameters if they were labeled and used for training. The intuition is
/// that samples which significantly change the model provide the most learning value.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>For each unlabeled sample, compute the expected gradient if it were labeled
/// with each possible class.</description></item>
/// <item><description>Weight each gradient by the probability of that class being correct.</description></item>
/// <item><description>The expected gradient magnitude indicates how much the model would change.</description></item>
/// <item><description>Select samples with the largest expected change.</description></item>
/// </list>
///
/// <para><b>Variants:</b></para>
/// <list type="bullet">
/// <item><description><b>EGL (Expected Gradient Length):</b> Uses the expected L2 norm of the gradient.</description></item>
/// <item><description><b>EMCM (Expected Model Change Maximization):</b> Uses gradient magnitude weighted by
/// current model confidence.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Settles et al., "Multiple-Instance Active Learning" (2008). ICML.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Multiple-Instance Active Learning", "https://dl.acm.org/doi/10.5555/1390156.1390273", Year = 2008, Authors = "Burr Settles, Mark Craven, Soumya Ray")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class ExpectedModelChange<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly ChangeMetric _metric;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines the change metric to use.
    /// </summary>
    public enum ChangeMetric
    {
        /// <summary>Expected L2 norm of the gradient.</summary>
        ExpectedGradientLength,
        /// <summary>Maximum gradient length across all possible labels.</summary>
        MaxGradientLength,
        /// <summary>Variance of gradient lengths across labels.</summary>
        GradientVariance
    }

    /// <summary>
    /// Initializes a new instance of the ExpectedModelChange class.
    /// </summary>
    /// <param name="metric">The change metric to use (default: ExpectedGradientLength).</param>
    public ExpectedModelChange(ChangeMetric metric = ChangeMetric.ExpectedGradientLength)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _metric = metric;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"ExpectedModelChange-{_metric}";

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
        var scores = ComputeExpectedChange(probs).ToVector();

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
    /// Computes the expected model change for every sample.
    /// </summary>
    /// <param name="probabilities">A [numSamples, numClasses] tensor of class probabilities.</param>
    private Tensor<T> ComputeExpectedChange(Tensor<T> probabilities)
    {
        var gradientLengths = ComputeGradientLengths(probabilities);
        return _metric switch
        {
            ChangeMetric.ExpectedGradientLength => ComputeEGL(probabilities, gradientLengths),
            ChangeMetric.MaxGradientLength => ActiveLearningEngineOps.MaxRows(gradientLengths),
            ChangeMetric.GradientVariance => ComputeGradientVariance(gradientLengths),
            _ => ComputeEGL(probabilities, gradientLengths)
        };
    }

    /// <summary>
    /// Computes the cross-entropy gradient length ||p - one_hot(y)|| for every sample and every
    /// possible label y.
    /// </summary>
    /// <returns>A [numSamples, numClasses] tensor; entry [i, y] = sqrt(Σ_c (p_ic - δ_cy)²).</returns>
    private static Tensor<T> ComputeGradientLengths(Tensor<T> probabilities)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = probabilities.Shape[0];
        int numClasses = probabilities.Shape[1];
        if (numSamples == 0 || numClasses == 0)
        {
            return new Tensor<T>(new[] { numSamples, numClasses });
        }

        var identity = engine.TensorEye<T>(numClasses).Reshape(new[] { 1, numClasses, numClasses });
        var blockRows = ActiveLearningEngineOps.BlockRows(numSamples, (long)numClasses * numClasses);
        var blocks = new List<Tensor<T>>();

        for (int start = 0; start < numSamples; start += blockRows)
        {
            int count = Math.Min(blockRows, numSamples - start);
            var shape = new[] { count, numClasses, numClasses };
            var block = engine.TensorSlice(probabilities, new[] { start, 0 }, new[] { count, numClasses })
                .Reshape(new[] { count, 1, numClasses });

            // Gradient components p_c - δ_{cy}, laid out as [sample, y, c]
            var gradComponents = engine.TensorSubtract(engine.TensorBroadcastTo(block, shape), engine.TensorBroadcastTo(identity, shape));
            var gradMagnitudeSq = engine.ReduceSum(engine.TensorMultiply(gradComponents, gradComponents), new[] { 2 }, keepDims: false);
            blocks.Add(engine.TensorSqrt(gradMagnitudeSq));
        }

        return engine.TensorConcatenate(blocks.ToArray(), 0);
    }

    /// <summary>
    /// Computes Expected Gradient Length (EGL).
    /// </summary>
    /// <remarks>
    /// For each possible label y, the gradient of cross-entropy loss is (p - one_hot(y)).
    /// The expected gradient length is: Σ_y P(y|x) * ||p - one_hot(y)||
    /// This simplifies to: Σ_y P(y|x) * sqrt(Σ_c (p_c - δ_{cy})²)
    /// </remarks>
    private static Tensor<T> ComputeEGL(Tensor<T> probabilities, Tensor<T> gradientLengths)
    {
        var engine = AiDotNetEngine.Current;
        if (probabilities.Shape[1] == 0)
        {
            return new Tensor<T>(new[] { probabilities.Shape[0] });
        }

        // Weight each label's gradient magnitude by the probability of that label
        return engine.ReduceSum(engine.TensorMultiply(probabilities, gradientLengths), new[] { 1 }, keepDims: false);
    }

    /// <summary>
    /// Computes the (population) variance of gradient lengths across possible labels.
    /// </summary>
    private Tensor<T> ComputeGradientVariance(Tensor<T> gradientLengths)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = gradientLengths.Shape[0];
        int numClasses = gradientLengths.Shape[1];
        var count = _numOps.FromDouble(numClasses);
        if (numSamples == 0 || numClasses == 0)
        {
            // Σ over no labels divided by zero labels, as the scalar formula gives
            return engine.TensorDivideScalar(new Tensor<T>(new[] { numSamples }), count);
        }

        var shape = new[] { numSamples, numClasses };
        var mean = engine.TensorDivideScalar(engine.ReduceSum(gradientLengths, new[] { 1 }, keepDims: true), count);
        var diff = engine.TensorSubtract(gradientLengths, engine.TensorBroadcastTo(mean, shape));
        var variance = engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 1 }, keepDims: false);
        return engine.TensorDivideScalar(variance, count);
    }

    /// <summary>
    /// Selects top-scoring samples without diversity consideration.
    /// </summary>
    private static int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Selects samples considering both expected change and diversity.
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
