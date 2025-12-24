using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Defines how the curriculum progresses over epochs.
/// </summary>
public enum CurriculumStrategy
{
    /// <summary>
    /// Linear progression from easy to all samples.
    /// </summary>
    Linear,

    /// <summary>
    /// Exponential progression (faster ramp-up of difficulty).
    /// </summary>
    Exponential,

    /// <summary>
    /// Step-wise progression with discrete difficulty levels.
    /// </summary>
    Stepped,

    /// <summary>
    /// Competence-based: difficulty threshold based on current performance.
    /// </summary>
    CompetenceBased
}

/// <summary>
/// A sampler that implements curriculum learning by progressively introducing harder samples.
/// </summary>
/// <typeparam name="T">The numeric type for difficulty scores.</typeparam>
/// <remarks>
/// <para>
/// CurriculumSampler implements the idea that models learn better when training starts
/// with easy examples and gradually progresses to harder ones, similar to how humans
/// learn complex subjects step by step.
/// </para>
/// <para><b>For Beginners:</b> Imagine teaching someone math: you start with 2+2,
/// not calculus. Curriculum learning applies this principle to machine learning:
///
/// - **Epoch 1**: Mostly easy samples (simple patterns, clear examples)
/// - **Epoch 5**: Mix of easy and medium samples
/// - **Epoch 10**: All samples including hard ones
///
/// This often leads to faster convergence and better final performance.
///
/// Example:
/// <code>
/// // Difficulty scores (0 = easy, 1 = hard)
/// var difficulties = samples.Select(s => ComputeDifficulty(s)).ToList();
/// var sampler = new CurriculumSampler&lt;float&gt;(difficulties, totalEpochs: 20);
///
/// for (int epoch = 0; epoch &lt; 20; epoch++)
/// {
///     sampler.OnEpochStart(epoch);
///     foreach (var idx in sampler.GetIndices())
///     {
///         // Earlier epochs sample more easy examples
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public class CurriculumSampler<T> : EpochAdaptiveSamplerBase<T>
{
    private readonly T[] _difficulties;
    private readonly CurriculumStrategy _strategy;
    private double _competence;

    /// <summary>
    /// Initializes a new instance of the CurriculumSampler class.
    /// </summary>
    /// <param name="difficulties">Difficulty score for each sample (0 = easiest, 1 = hardest).</param>
    /// <param name="totalEpochs">Total number of epochs for curriculum completion.</param>
    /// <param name="strategy">The curriculum progression strategy.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public CurriculumSampler(
        IEnumerable<T> difficulties,
        int totalEpochs,
        CurriculumStrategy strategy = CurriculumStrategy.Linear,
        int? seed = null)
        : base(totalEpochs, seed)
    {
        _difficulties = difficulties?.ToArray() ?? throw new ArgumentNullException(nameof(difficulties));
        _strategy = strategy;
        _competence = 0.0;
    }

    /// <inheritdoc/>
    public override int Length => _difficulties.Length;

    /// <summary>
    /// Gets the current difficulty threshold based on epoch and strategy.
    /// </summary>
    public double CurrentDifficultyThreshold
    {
        get
        {
            double progress = Progress;

            return _strategy switch
            {
                CurriculumStrategy.Linear => progress,
                CurriculumStrategy.Exponential => Math.Pow(progress, 2),
                CurriculumStrategy.Stepped => GetSteppedThreshold(progress),
                CurriculumStrategy.CompetenceBased => _competence,
                _ => progress
            };
        }
    }

    /// <summary>
    /// Sets the model's current competence level for competence-based curriculum.
    /// </summary>
    /// <param name="competence">Competence level from 0 (beginner) to 1 (expert).</param>
    public void SetCompetence(double competence)
    {
        _competence = Math.Max(0.0, Math.Min(1.0, competence));
    }

    private static double GetSteppedThreshold(double progress)
    {
        // 5 discrete steps
        if (progress < 0.2) return 0.2;
        if (progress < 0.4) return 0.4;
        if (progress < 0.6) return 0.6;
        if (progress < 0.8) return 0.8;
        return 1.0;
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        double threshold = CurrentDifficultyThreshold;

        // Filter samples that are below the current difficulty threshold
        var eligibleIndices = new List<int>();
        for (int i = 0; i < _difficulties.Length; i++)
        {
            double difficulty = NumOps.ToDouble(_difficulties[i]);
            if (difficulty <= threshold)
            {
                eligibleIndices.Add(i);
            }
        }

        // If no samples are eligible (threshold too low), include easiest samples
        if (eligibleIndices.Count == 0)
        {
            // Find minimum difficulty and include all samples at that level
            double minDifficulty = double.MaxValue;
            for (int i = 0; i < _difficulties.Length; i++)
            {
                double d = NumOps.ToDouble(_difficulties[i]);
                if (d < minDifficulty) minDifficulty = d;
            }

            for (int i = 0; i < _difficulties.Length; i++)
            {
                if (Math.Abs(NumOps.ToDouble(_difficulties[i]) - minDifficulty) < 1e-9)
                {
                    eligibleIndices.Add(i);
                }
            }
        }

        // Shuffle eligible indices
        int[] shuffled = eligibleIndices.ToArray();
        ShuffleIndices(shuffled);

        foreach (int idx in shuffled)
        {
            yield return idx;
        }
    }
}

/// <summary>
/// A sampler that implements self-paced learning with automatic difficulty adjustment.
/// </summary>
/// <typeparam name="T">The numeric type for losses and weights.</typeparam>
/// <remarks>
/// <para>
/// SelfPacedSampler automatically adjusts sample selection based on the model's
/// performance on each sample. Samples with lower loss (easier for the model)
/// are more likely to be selected early, with harder samples gradually introduced.
/// </para>
/// <para><b>For Beginners:</b> Unlike CurriculumSampler where YOU define difficulty,
/// SelfPacedSampler lets the MODEL decide what's easy based on its own performance:
///
/// - Samples with low loss = easy = selected early
/// - Samples with high loss = hard = selected later
///
/// This is adaptive curriculum learning - the curriculum adjusts based on the model!
/// </para>
/// </remarks>
public class SelfPacedSampler<T> : EpochAdaptiveSamplerBase<T>
{
    private readonly T[] _losses;
    private T _lambda; // Pace parameter
    private readonly T _lambdaGrowthRate;

    /// <summary>
    /// Initializes a new instance of the SelfPacedSampler class.
    /// </summary>
    /// <param name="datasetSize">The total number of samples.</param>
    /// <param name="initialLambda">Initial pace parameter (lower = stricter selection).</param>
    /// <param name="lambdaGrowthRate">How much lambda increases each epoch.</param>
    /// <param name="totalEpochs">Total epochs for training (used for progress tracking).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SelfPacedSampler(
        int datasetSize,
        T initialLambda,
        T lambdaGrowthRate,
        int totalEpochs = 100,
        int? seed = null)
        : base(totalEpochs, seed)
    {
        _losses = new T[datasetSize];
        for (int i = 0; i < datasetSize; i++)
        {
            _losses[i] = NumOps.Zero; // Initialize with zero loss (will be updated during training)
        }

        _lambda = initialLambda;
        _lambdaGrowthRate = lambdaGrowthRate;
    }

    /// <inheritdoc/>
    public override int Length => _losses.Length;

    /// <summary>
    /// Gets the current pace parameter lambda.
    /// </summary>
    public T Lambda => _lambda;

    /// <summary>
    /// Updates the loss for a specific sample.
    /// </summary>
    /// <param name="index">The sample index.</param>
    /// <param name="loss">The current loss for this sample.</param>
    public void UpdateLoss(int index, T loss)
    {
        if (index >= 0 && index < _losses.Length)
        {
            _losses[index] = loss;
        }
    }

    /// <summary>
    /// Batch updates losses for multiple samples.
    /// </summary>
    /// <param name="indices">The sample indices.</param>
    /// <param name="losses">The losses for each sample.</param>
    public void UpdateLosses(IReadOnlyList<int> indices, IReadOnlyList<T> losses)
    {
        for (int i = 0; i < indices.Count && i < losses.Count; i++)
        {
            UpdateLoss(indices[i], losses[i]);
        }
    }

    /// <inheritdoc/>
    protected override void OnEpochStartCore(int epoch)
    {
        // Increase lambda for next epoch
        _lambda = NumOps.Add(_lambda, _lambdaGrowthRate);
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        double lambdaValue = NumOps.ToDouble(_lambda);

        // Compute selection weight for each sample
        // Weight = 1 if loss < lambda, 0 otherwise (hard thresholding)
        var eligibleIndices = new List<int>();
        for (int i = 0; i < _losses.Length; i++)
        {
            double loss = NumOps.ToDouble(_losses[i]);
            if (loss <= lambdaValue)
            {
                eligibleIndices.Add(i);
            }
        }

        // If too few samples are eligible, include more
        if (eligibleIndices.Count < _losses.Length / 10) // At least 10%
        {
            // Sort by loss and take the easiest samples
            var sortedIndices = Enumerable.Range(0, _losses.Length)
                .OrderBy(i => NumOps.ToDouble(_losses[i]))
                .Take(_losses.Length / 10 + 1)
                .ToList();
            eligibleIndices = sortedIndices;
        }

        // Shuffle eligible indices
        int[] shuffled = eligibleIndices.ToArray();
        ShuffleIndices(shuffled);

        foreach (int idx in shuffled)
        {
            yield return idx;
        }
    }
}
