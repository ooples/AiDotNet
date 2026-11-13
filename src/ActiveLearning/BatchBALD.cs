using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements BatchBALD for joint batch selection in active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BatchBALD extends BALD to select batches of samples that are
/// jointly informative. Unlike naive greedy selection that picks individual high-BALD samples,
/// BatchBALD considers the joint mutual information to avoid redundant selections.</para>
///
/// <para><b>Problem with naive BALD:</b> Selecting top-k samples by individual BALD scores
/// may result in redundant samples that provide similar information.</para>
///
/// <para><b>Solution:</b> BatchBALD computes joint mutual information for candidate batches:
/// I(y₁, y₂, ..., yₖ; θ|x₁, x₂, ..., xₖ, D)</para>
///
/// <para><b>Algorithm (Greedy Approximation):</b></para>
/// <list type="number">
/// <item><description>Start with empty batch.</description></item>
/// <item><description>For each candidate, compute gain in joint mutual information if added.</description></item>
/// <item><description>Add candidate with highest marginal gain to batch.</description></item>
/// <item><description>Repeat until batch is full.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Kirsch, A., van Amersfoort, J., &amp; Gal, Y. (2019). "BatchBALD:
/// Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning."</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning", "https://arxiv.org/abs/1906.08158", Year = 2019, Authors = "Andreas Kirsch, Joost van Amersfoort, Yarin Gal")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class BatchBALD<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numMcSamples;
    private readonly double _dropoutRate;
    private readonly int _candidatePoolSize;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the BatchBALD class.
    /// </summary>
    /// <param name="numMcSamples">Number of MC samples for Bayesian approximation (default: 10).</param>
    /// <param name="dropoutRate">Dropout rate for MC Dropout (default: 0.5).</param>
    /// <param name="candidatePoolSize">Size of candidate pool for greedy selection (default: 100).
    /// Larger values are more accurate but slower.</param>
    public BatchBALD(int numMcSamples = 10, double dropoutRate = 0.5, int candidatePoolSize = 100)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numMcSamples = numMcSamples;
        _dropoutRate = dropoutRate;
        _candidatePoolSize = candidatePoolSize;
        _useBatchDiversity = true; // BatchBALD inherently considers batch diversity
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"BatchBALD-MC{_numMcSamples}";

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

        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        // Get MC predictions
        var mcPredictions = GetMCPredictions(model, unlabeledPool);
        var numClasses = mcPredictions[0].Length / numSamples;

        // Precompute individual BALD scores for candidate selection
        var mcProbs = ActiveLearningEngineOps.SoftmaxAll(mcPredictions, numSamples, numClasses);
        var baldScores = ActiveLearningEngineOps.BaldScores(mcProbs, numSamples).ToVector();

        // Select top candidates for batch selection
        var candidateIndices = GetTopCandidates(baldScores, Math.Min(_candidatePoolSize, numSamples));

        // Greedy batch selection using joint mutual information
        var selectedBatch = GreedyBatchSelection(mcProbs, baldScores, candidateIndices, batchSize, numClasses);

        UpdateStatistics(baldScores);
        return selectedBatch;
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        var mcPredictions = GetMCPredictions(model, unlabeledPool);
        var numClasses = mcPredictions[0].Length / numSamples;

        // Return individual BALD scores for informativeness
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
    /// Gets MC predictions by running multiple forward passes.
    /// </summary>
    private List<Tensor<T>> GetMCPredictions(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> pool)
    {
        var mcPredictions = new List<Tensor<T>>();
        for (int m = 0; m < _numMcSamples; m++)
        {
            var predictions = model.Predict(pool);
            mcPredictions.Add(AddDropoutNoise(predictions, _dropoutRate, m));
        }
        return mcPredictions;
    }

    /// <summary>
    /// Adds simulated dropout noise to predictions.
    /// </summary>
    private Tensor<T> AddDropoutNoise(Tensor<T> predictions, double dropoutRate, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var noisyPredictions = new Tensor<T>(predictions._shape);

        for (int i = 0; i < predictions.Length; i++)
        {
            var value = _numOps.ToDouble(predictions[i]);

            // Simulate MC Dropout: randomly zero neurons and scale survivors
            if (random.NextDouble() < dropoutRate)
            {
                // Drop this neuron (set to zero)
                value = 0.0;
            }
            else
            {
                // Scale up surviving values to maintain expected value (inverted dropout)
                value *= 1.0 / (1.0 - dropoutRate);
            }

            noisyPredictions[i] = _numOps.FromDouble(value);
        }

        return noisyPredictions;
    }

    /// <summary>
    /// Performs greedy batch selection based on joint mutual information.
    /// </summary>
    /// <remarks>
    /// The joint MI of a batch is approximated as Σ_i BALD_i - Σ_{i&lt;j} R(i, j), with pairwise
    /// redundancy R(i, j) = 0.1 × mean_m ⟨p_m(i), p_m(j)⟩. The marginal gain of a candidate c is
    /// therefore BALD_c - Σ_{s ∈ batch} R(c, s), which is accumulated as samples are selected.
    /// </remarks>
    private int[] GreedyBatchSelection(List<Tensor<T>> mcProbs, Vector<T> baldScores, int[] candidates, int batchSize, int numClasses)
    {
        var engine = AiDotNetEngine.Current;
        var selected = new List<int>();
        int numCandidates = candidates.Length;
        if (numCandidates == 0 || batchSize <= 0)
        {
            return [.. selected];
        }

        var candidateIndices = new Tensor<int>(candidates, new[] { numCandidates });

        // Pairwise redundancy between candidates from their average prediction agreement
        var agreement = new Tensor<T>(new[] { numCandidates, numCandidates });
        if (numClasses > 0)
        {
            foreach (var probs in mcProbs)
            {
                var candidateProbs = engine.TensorIndexSelect(probs, candidateIndices, 0);
                agreement = engine.TensorAdd(agreement, engine.TensorMatMul(candidateProbs, engine.TensorTranspose(candidateProbs)));
            }
        }

        agreement = engine.TensorDivideScalar(agreement, _numOps.FromDouble(mcProbs.Count));

        // Higher agreement = more redundancy; scale to be a small fraction of MI
        var redundancy = engine.TensorMultiplyScalar(agreement, _numOps.FromDouble(0.1));

        // First sample: gain is the individual BALD score
        var individual = engine.TensorIndexSelect(Tensor<T>.FromVector(baldScores), candidateIndices, 0)
            .Reshape(new[] { 1, numCandidates });
        var batchRedundancy = new Tensor<T>(new[] { 1, numCandidates });
        var excluded = new bool[numCandidates];

        while (selected.Count < batchSize && selected.Count < numCandidates)
        {
            var gains = selected.Count == 0 ? individual : engine.TensorSubtract(individual, batchRedundancy);
            var best = ActiveLearningEngineOps.ArgMaxAbove(gains, excluded, _numOps.MinValue);
            if (best < 0)
            {
                break;
            }

            selected.Add(candidates[best]);
            excluded[best] = true;
            var bestRedundancy = engine.TensorSlice(redundancy, new[] { best, 0 }, new[] { 1, numCandidates });
            batchRedundancy = engine.TensorAdd(batchRedundancy, bestRedundancy);
        }

        return [.. selected];
    }

    /// <summary>
    /// Gets top candidates based on individual BALD scores.
    /// </summary>
    private static int[] GetTopCandidates(Vector<T> scores, int topK)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, topK);
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
