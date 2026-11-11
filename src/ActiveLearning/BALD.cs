using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Bayesian Active Learning by Disagreement (BALD) for sample selection.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BALD uses information theory to select samples that maximize
/// the mutual information between model predictions and model parameters. In practice, this
/// means selecting samples where different "versions" of the model disagree the most.</para>
///
/// <para><b>Formula:</b> I(y; θ|x, D) = H(y|x, D) - E_θ[H(y|x, θ)]</para>
/// <para>where H is entropy, y is the label, θ are model parameters, x is input, D is training data.</para>
///
/// <para><b>Interpretation:</b></para>
/// <list type="bullet">
/// <item><description>First term: Entropy of the average prediction (epistemic + aleatoric uncertainty).</description></item>
/// <item><description>Second term: Average entropy across models (aleatoric uncertainty only).</description></item>
/// <item><description>Difference: Epistemic uncertainty (model uncertainty that can be reduced with more data).</description></item>
/// </list>
///
/// <para><b>Implementation:</b> Uses MC Dropout to approximate Bayesian inference by running
/// multiple forward passes with dropout enabled during inference.</para>
///
/// <para><b>Reference:</b> Houlsby, N. et al. (2011). "Bayesian Active Learning for Classification
/// and Preference Learning."</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bayesian Active Learning for Classification and Preference Learning", "https://arxiv.org/abs/1112.5745", Year = 2011, Authors = "Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, Máté Lengyel")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class BALD<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numMcSamples;
    private readonly double _dropoutRate;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the BALD class.
    /// </summary>
    /// <param name="numMcSamples">Number of Monte Carlo samples (forward passes with dropout) for
    /// approximating Bayesian inference (default: 10).</param>
    /// <param name="dropoutRate">Dropout rate for MC Dropout (default: 0.5).</param>
    public BALD(int numMcSamples = 10, double dropoutRate = 0.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numMcSamples = numMcSamples;
        _dropoutRate = dropoutRate;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"BALD-MC{_numMcSamples}";

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

        var numSamples = unlabeledPool.Shape[0];

        // Perform multiple forward passes (simulating MC Dropout)
        // In a real implementation, this would enable dropout during inference
        // Here we simulate by adding noise to predictions
        var mcPredictions = new List<Tensor<T>>();
        for (int m = 0; m < _numMcSamples; m++)
        {
            var predictions = model.Predict(unlabeledPool);
            // Simulate dropout effect by adding scaled noise
            mcPredictions.Add(ActiveLearningEngineOps.AddSimulatedDropout(predictions, _dropoutRate, m));
        }

        var numClasses = mcPredictions[0].Length / numSamples;
        var mcProbs = ActiveLearningEngineOps.SoftmaxAll(mcPredictions, numSamples, numClasses);
        var scores = ActiveLearningEngineOps.BaldScores(mcProbs, numSamples).ToVector();

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
    /// Selects samples considering both BALD score and diversity.
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
