using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// A sampler that implements importance sampling for variance reduction.
/// </summary>
/// <typeparam name="T">The numeric type for importance weights.</typeparam>
/// <remarks>
/// <para>
/// ImportanceSampler samples data points based on their importance, typically
/// computed from gradient norms, loss values, or uncertainty estimates.
/// This can accelerate training by focusing on samples that contribute most to learning.
/// </para>
/// <para><b>For Beginners:</b> Not all training samples are equally useful.
/// Importance sampling focuses training on the most informative samples:
///
/// - **High gradient norm** = Sample provides strong learning signal
/// - **High loss** = Model is uncertain, needs more training
/// - **High uncertainty** = Model needs to see this more
///
/// This can reduce training time by 2-3x compared to uniform sampling!
///
/// Example:
/// <code>
/// var sampler = new ImportanceSampler&lt;float&gt;(datasetSize: 1000);
///
/// // After each batch, update importance based on gradient norms
/// foreach (var (idx, gradNorm) in batch.Zip(gradientNorms))
/// {
///     sampler.UpdateImportance(idx, gradNorm);
/// }
/// </code>
/// </para>
/// </remarks>
public class ImportanceSampler<T> : DataSamplerBase
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private T[] _importanceScores;
    private double[] _cumulativeProbabilities;
    private readonly double _smoothingFactor;
    private readonly bool _stabilize;
    private bool _needsUpdate = true;

    /// <summary>
    /// Initializes a new instance of the ImportanceSampler class.
    /// </summary>
    /// <param name="datasetSize">The total number of samples.</param>
    /// <param name="smoothingFactor">Smoothing factor to prevent extreme sampling (0.1-0.5 recommended).</param>
    /// <param name="stabilize">Whether to clip extreme importance values.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ImportanceSampler(
        int datasetSize,
        double smoothingFactor = 0.2,
        bool stabilize = true,
        int? seed = null)
        : base(seed)
    {
        if (datasetSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(datasetSize), "Dataset size must be at least 1.");
        }

        _importanceScores = new T[datasetSize];
        _cumulativeProbabilities = new double[datasetSize];

        // Initialize with uniform importance
        T initialImportance = NumOps.FromDouble(1.0 / datasetSize);
        for (int i = 0; i < datasetSize; i++)
        {
            _importanceScores[i] = initialImportance;
        }

        _smoothingFactor = Math.Max(0.0, Math.Min(1.0, smoothingFactor));
        _stabilize = stabilize;
    }

    /// <inheritdoc/>
    public override int Length => _importanceScores.Length;

    /// <summary>
    /// Gets the importance scores for all samples.
    /// </summary>
    public IReadOnlyList<T> ImportanceScores => _importanceScores;

    /// <summary>
    /// Updates the importance score for a single sample.
    /// </summary>
    /// <param name="index">The sample index.</param>
    /// <param name="importance">The new importance score.</param>
    public void UpdateImportance(int index, T importance)
    {
        if (index >= 0 && index < _importanceScores.Length)
        {
            _importanceScores[index] = importance;
            _needsUpdate = true;
        }
    }

    /// <summary>
    /// Batch updates importance scores.
    /// </summary>
    /// <param name="indices">The sample indices.</param>
    /// <param name="importances">The importance scores.</param>
    public void UpdateImportances(IReadOnlyList<int> indices, IReadOnlyList<T> importances)
    {
        for (int i = 0; i < indices.Count && i < importances.Count; i++)
        {
            if (indices[i] >= 0 && indices[i] < _importanceScores.Length)
            {
                _importanceScores[indices[i]] = importances[i];
            }
        }
        _needsUpdate = true;
    }

    /// <summary>
    /// Sets all importance scores at once.
    /// </summary>
    /// <param name="importances">Array of importance scores.</param>
    public void SetImportances(IReadOnlyList<T> importances)
    {
        if (importances.Count != _importanceScores.Length)
        {
            throw new ArgumentException($"Expected {_importanceScores.Length} importance scores, got {importances.Count}.");
        }

        for (int i = 0; i < _importanceScores.Length; i++)
        {
            _importanceScores[i] = importances[i];
        }
        _needsUpdate = true;
    }

    /// <summary>
    /// Recomputes the cumulative probability distribution.
    /// </summary>
    private void RecomputeProbabilities()
    {
        if (!_needsUpdate) return;

        int n = _importanceScores.Length;
        double[] rawScores = new double[n];

        // Convert to doubles
        for (int i = 0; i < n; i++)
        {
            rawScores[i] = Math.Max(0.0, NumOps.ToDouble(_importanceScores[i]));
        }

        // Stabilization: clip extreme values
        if (_stabilize)
        {
            double mean = rawScores.Average();
            double stdDev = Math.Sqrt(rawScores.Average(x => Math.Pow(x - mean, 2)));
            double maxAllowed = mean + 3 * stdDev;

            for (int i = 0; i < n; i++)
            {
                rawScores[i] = Math.Min(rawScores[i], maxAllowed);
            }
        }

        // Add smoothing: blend with uniform distribution
        double uniformProb = 1.0 / n;
        for (int i = 0; i < n; i++)
        {
            rawScores[i] = (1.0 - _smoothingFactor) * rawScores[i] + _smoothingFactor * uniformProb;
        }

        // Normalize
        double sum = rawScores.Sum();
        if (sum <= 0)
        {
            // Fallback to uniform
            for (int i = 0; i < n; i++)
            {
                rawScores[i] = uniformProb;
            }
            sum = 1.0;
        }

        // Compute cumulative probabilities
        double cumulative = 0;
        for (int i = 0; i < n; i++)
        {
            cumulative += rawScores[i] / sum;
            _cumulativeProbabilities[i] = cumulative;
        }
        _cumulativeProbabilities[n - 1] = 1.0; // Ensure exactly 1.0

        _needsUpdate = false;
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        RecomputeProbabilities();

        // Sample with replacement based on importance
        for (int i = 0; i < _importanceScores.Length; i++)
        {
            yield return SampleOne();
        }
    }

    /// <summary>
    /// Gets importance-weighted sample indices without replacement.
    /// </summary>
    /// <param name="count">Number of samples to draw.</param>
    /// <returns>Sampled indices.</returns>
    public IEnumerable<int> GetIndicesWithoutReplacement(int count)
    {
        RecomputeProbabilities();

        var selected = new HashSet<int>();
        int maxAttempts = count * 10;
        int attempts = 0;

        while (selected.Count < count && selected.Count < _importanceScores.Length && attempts < maxAttempts)
        {
            int idx = SampleOne();
            if (selected.Add(idx))
            {
                yield return idx;
            }
            attempts++;
        }

        // If we haven't gotten enough, add remaining indices
        if (selected.Count < count)
        {
            for (int i = 0; i < _importanceScores.Length && selected.Count < count; i++)
            {
                if (selected.Add(i))
                {
                    yield return i;
                }
            }
        }
    }

    private int SampleOne()
    {
        double u = Random.NextDouble();

        // Binary search
        int left = 0;
        int right = _cumulativeProbabilities.Length - 1;

        while (left < right)
        {
            int mid = (left + right) / 2;
            if (_cumulativeProbabilities[mid] < u)
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }

        return left;
    }

    /// <summary>
    /// Gets the sampling weight correction factor for a sample.
    /// </summary>
    /// <param name="index">The sample index.</param>
    /// <returns>The correction factor to apply to gradients for unbiased estimation.</returns>
    /// <remarks>
    /// When using importance sampling, gradients should be corrected by 1/p(i)
    /// where p(i) is the probability of sampling index i. This ensures unbiased
    /// gradient estimates despite non-uniform sampling.
    /// </remarks>
    public T GetCorrectionFactor(int index)
    {
        RecomputeProbabilities();

        double prob = index == 0
            ? _cumulativeProbabilities[0]
            : _cumulativeProbabilities[index] - _cumulativeProbabilities[index - 1];

        // Correction = 1 / (n * p_i) where n is dataset size
        double correction = 1.0 / (_importanceScores.Length * Math.Max(prob, 1e-10));
        return NumOps.FromDouble(correction);
    }
}

/// <summary>
/// Active learning selection strategies.
/// </summary>
public enum ActiveLearningStrategy
{
    /// <summary>
    /// Select samples with highest uncertainty (e.g., entropy, margin).
    /// </summary>
    Uncertainty,

    /// <summary>
    /// Select diverse samples using distance-based clustering.
    /// </summary>
    Diversity,

    /// <summary>
    /// Combine uncertainty and diversity.
    /// </summary>
    Hybrid,

    /// <summary>
    /// Random sampling (baseline).
    /// </summary>
    Random
}

/// <summary>
/// A sampler for active learning that selects the most informative samples for labeling.
/// </summary>
/// <typeparam name="T">The numeric type for uncertainty scores.</typeparam>
/// <remarks>
/// <para>
/// ActiveLearningSampler implements uncertainty sampling and other active learning
/// strategies to select unlabeled samples that would be most valuable to label.
/// </para>
/// <para><b>For Beginners:</b> In active learning, you don't have labels for all data.
/// This sampler helps you decide which unlabeled samples to ask an expert to label:
///
/// - **Uncertainty sampling**: Ask about samples the model is unsure about
/// - **Diversity sampling**: Ask about samples that are different from what you've seen
/// - **Expected model change**: Ask about samples that would change the model most
///
/// This can dramatically reduce labeling costs (50-90% less labels needed)!
/// </para>
/// </remarks>
public class ActiveLearningSampler<T> : DataSamplerBase
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T[] _uncertaintyScores;
    private readonly bool[] _isLabeled;
    private readonly ActiveLearningStrategy _strategy;
    private readonly double _diversityWeight;

    /// <summary>
    /// Initializes a new instance of the ActiveLearningSampler class.
    /// </summary>
    /// <param name="datasetSize">The total number of samples.</param>
    /// <param name="strategy">The active learning selection strategy.</param>
    /// <param name="diversityWeight">Weight for diversity in hybrid strategy (0-1).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ActiveLearningSampler(
        int datasetSize,
        ActiveLearningStrategy strategy = ActiveLearningStrategy.Uncertainty,
        double diversityWeight = 0.3,
        int? seed = null)
        : base(seed)
    {
        if (datasetSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(datasetSize), "Dataset size must be at least 1.");
        }

        _uncertaintyScores = new T[datasetSize];
        _isLabeled = new bool[datasetSize];
        _strategy = strategy;
        _diversityWeight = Math.Max(0.0, Math.Min(1.0, diversityWeight));

        // Initialize with uniform uncertainty
        T initialUncertainty = NumOps.FromDouble(0.5);
        for (int i = 0; i < datasetSize; i++)
        {
            _uncertaintyScores[i] = initialUncertainty;
        }
    }

    /// <inheritdoc/>
    public override int Length => _uncertaintyScores.Length;

    /// <summary>
    /// Gets the number of labeled samples.
    /// </summary>
    public int LabeledCount => _isLabeled.Count(x => x);

    /// <summary>
    /// Gets the number of unlabeled samples.
    /// </summary>
    public int UnlabeledCount => _isLabeled.Count(x => !x);

    /// <summary>
    /// Marks a sample as labeled.
    /// </summary>
    /// <param name="index">The sample index.</param>
    public void MarkAsLabeled(int index)
    {
        if (index >= 0 && index < _isLabeled.Length)
        {
            _isLabeled[index] = true;
        }
    }

    /// <summary>
    /// Marks multiple samples as labeled.
    /// </summary>
    /// <param name="indices">The sample indices.</param>
    public void MarkAsLabeled(IEnumerable<int> indices)
    {
        foreach (int idx in indices)
        {
            MarkAsLabeled(idx);
        }
    }

    /// <summary>
    /// Updates the uncertainty score for a sample.
    /// </summary>
    /// <param name="index">The sample index.</param>
    /// <param name="uncertainty">The uncertainty score (higher = more uncertain).</param>
    public void UpdateUncertainty(int index, T uncertainty)
    {
        if (index >= 0 && index < _uncertaintyScores.Length)
        {
            _uncertaintyScores[index] = uncertainty;
        }
    }

    /// <summary>
    /// Batch updates uncertainty scores.
    /// </summary>
    /// <param name="indices">The sample indices.</param>
    /// <param name="uncertainties">The uncertainty scores.</param>
    public void UpdateUncertainties(IReadOnlyList<int> indices, IReadOnlyList<T> uncertainties)
    {
        for (int i = 0; i < indices.Count && i < uncertainties.Count; i++)
        {
            UpdateUncertainty(indices[i], uncertainties[i]);
        }
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        // For active learning, we typically only return labeled samples for training
        var labeledIndices = new List<int>();
        for (int i = 0; i < _isLabeled.Length; i++)
        {
            if (_isLabeled[i])
            {
                labeledIndices.Add(i);
            }
        }

        // Shuffle labeled indices
        int[] shuffled = labeledIndices.ToArray();
        ShuffleIndices(shuffled);

        foreach (int idx in shuffled)
        {
            yield return idx;
        }
    }

    /// <summary>
    /// Selects the most informative unlabeled samples for labeling.
    /// </summary>
    /// <param name="numToSelect">Number of samples to select.</param>
    /// <returns>Indices of selected samples.</returns>
    public IEnumerable<int> SelectForLabeling(int numToSelect)
    {
        var unlabeledIndices = new List<int>();
        for (int i = 0; i < _isLabeled.Length; i++)
        {
            if (!_isLabeled[i])
            {
                unlabeledIndices.Add(i);
            }
        }

        if (unlabeledIndices.Count == 0)
        {
            yield break;
        }

        numToSelect = Math.Min(numToSelect, unlabeledIndices.Count);

        switch (_strategy)
        {
            case ActiveLearningStrategy.Uncertainty:
                foreach (int idx in SelectByUncertainty(unlabeledIndices, numToSelect))
                {
                    yield return idx;
                }
                break;

            case ActiveLearningStrategy.Diversity:
                foreach (int idx in SelectByDiversity(unlabeledIndices, numToSelect))
                {
                    yield return idx;
                }
                break;

            case ActiveLearningStrategy.Hybrid:
                foreach (int idx in SelectHybrid(unlabeledIndices, numToSelect))
                {
                    yield return idx;
                }
                break;

            case ActiveLearningStrategy.Random:
            default:
                // Shuffle and take first N
                int[] shuffled = unlabeledIndices.ToArray();
                ShuffleIndices(shuffled);
                for (int i = 0; i < numToSelect; i++)
                {
                    yield return shuffled[i];
                }
                break;
        }
    }

    private IEnumerable<int> SelectByUncertainty(List<int> candidates, int numToSelect)
    {
        // Sort by uncertainty (descending) and take top N
        return candidates
            .OrderByDescending(i => NumOps.ToDouble(_uncertaintyScores[i]))
            .Take(numToSelect);
    }

    private IEnumerable<int> SelectByDiversity(List<int> candidates, int numToSelect)
    {
        // Simple diversity: spread selection evenly across the candidates
        // A more sophisticated version would use feature-based clustering
        var selected = new List<int>();
        int step = Math.Max(1, candidates.Count / numToSelect);

        for (int i = 0; i < candidates.Count && selected.Count < numToSelect; i += step)
        {
            selected.Add(candidates[i]);
        }

        // If we need more, take from remaining
        int idx = 0;
        while (selected.Count < numToSelect && idx < candidates.Count)
        {
            if (!selected.Contains(candidates[idx]))
            {
                selected.Add(candidates[idx]);
            }
            idx++;
        }

        return selected;
    }

    private IEnumerable<int> SelectHybrid(List<int> candidates, int numToSelect)
    {
        // Combine uncertainty and diversity
        int uncertaintyCount = (int)((1 - _diversityWeight) * numToSelect);
        int diversityCount = numToSelect - uncertaintyCount;

        var selected = new HashSet<int>();

        // First, select by uncertainty
        foreach (int idx in SelectByUncertainty(candidates, uncertaintyCount))
        {
            selected.Add(idx);
        }

        // Then, select diverse samples from remaining
        var remaining = candidates.Where(i => !selected.Contains(i)).ToList();
        foreach (int idx in SelectByDiversity(remaining, diversityCount))
        {
            selected.Add(idx);
        }

        return selected;
    }
}
