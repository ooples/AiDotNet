using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Base class for all data samplers providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// DataSamplerBase provides default implementations for common sampler operations
/// like random seed management and epoch callbacks. All concrete samplers should
/// inherit from this base class rather than implementing IDataSampler directly.
/// </para>
/// <para><b>For Beginners:</b> This base class handles the common plumbing that all
/// samplers need, like managing random number generators for reproducibility.
/// When creating a custom sampler, inherit from this class and override GetIndicesCore().
/// </para>
/// </remarks>
public abstract class DataSamplerBase : IDataSampler
{
    /// <summary>
    /// The random number generator used for sampling.
    /// </summary>
    protected Random Random;

    /// <summary>
    /// The current epoch number (0-based).
    /// </summary>
    protected int CurrentEpoch;

    /// <summary>
    /// Initializes a new instance of the DataSamplerBase class.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected DataSamplerBase(int? seed = null)
    {
        Random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        CurrentEpoch = 0;
    }

    /// <inheritdoc/>
    public abstract int Length { get; }

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        return GetIndicesCore();
    }

    /// <summary>
    /// Core implementation for generating indices. Override this in derived classes.
    /// </summary>
    /// <returns>An enumerable of sample indices.</returns>
    protected abstract IEnumerable<int> GetIndicesCore();

    /// <inheritdoc/>
    public virtual void SetSeed(int seed)
    {
        Random = RandomHelper.CreateSeededRandom(seed);
    }

    /// <inheritdoc/>
    public virtual void OnEpochStart(int epoch)
    {
        CurrentEpoch = epoch;
    }

    /// <summary>
    /// Performs Fisher-Yates shuffle on an array of indices.
    /// </summary>
    /// <param name="indices">The array to shuffle in-place.</param>
    protected void ShuffleIndices(int[] indices)
    {
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = Random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }

    /// <summary>
    /// Creates a sequential array of indices from 0 to count-1.
    /// </summary>
    /// <param name="count">The number of indices to create.</param>
    /// <returns>An array containing indices [0, 1, 2, ..., count-1].</returns>
    protected static int[] CreateSequentialIndices(int count)
    {
        int[] indices = new int[count];
        for (int i = 0; i < count; i++)
        {
            indices[i] = i;
        }
        return indices;
    }
}

/// <summary>
/// Base class for weighted samplers providing common weight-based functionality.
/// </summary>
/// <typeparam name="T">The numeric type for weights.</typeparam>
/// <remarks>
/// <para>
/// WeightedSamplerBase provides common functionality for samplers that use per-sample
/// weights, including cumulative probability computation and weighted random selection.
/// </para>
/// </remarks>
public abstract class WeightedSamplerBase<T> : DataSamplerBase, IWeightedSampler<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The weights for each sample.
    /// </summary>
    protected T[] WeightsArray;

    /// <summary>
    /// Cumulative probabilities for weighted sampling.
    /// </summary>
    /// <remarks>
    /// Kept in double whatever <typeparamref name="T"/> is: a running sum of probabilities in single or half
    /// precision drifts (and in half precision saturates), which skews every draw toward the last index.
    /// </remarks>
    protected Tensor<double> CumulativeProbabilities;

    /// <summary>
    /// Whether to sample with replacement.
    /// </summary>
    protected bool ReplacementEnabled;

    /// <summary>
    /// Number of samples to draw per epoch.
    /// </summary>
    protected int? NumSamplesOverride;

    /// <summary>
    /// Initializes a new instance of the WeightedSamplerBase class.
    /// </summary>
    /// <param name="weights">The weight for each sample.</param>
    /// <param name="numSamples">Number of samples to draw per epoch.</param>
    /// <param name="replacement">Whether to sample with replacement.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected WeightedSamplerBase(
        IEnumerable<T> weights,
        int? numSamples = null,
        bool replacement = true,
        int? seed = null)
        : base(seed)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        WeightsArray = weights.ToArray();
        if (WeightsArray.Length == 0)
        {
            throw new ArgumentException("Weights cannot be empty.", nameof(weights));
        }

        NumSamplesOverride = numSamples;
        ReplacementEnabled = replacement;
        CumulativeProbabilities = new Tensor<double>(new[] { WeightsArray.Length });
        // Call the private core method directly to avoid virtual call in constructor
        ComputeCumulativeProbabilitiesCore();
    }

    /// <inheritdoc/>
    public override int Length => NumSamplesOverride ?? WeightsArray.Length;

    /// <inheritdoc/>
    public IReadOnlyList<T> Weights
    {
        get => WeightsArray;
        set
        {
            Guard.NotNull(value);
            var array = value.ToArray();
            if (array.Length == 0)
            {
                throw new ArgumentException("Weights cannot be empty.", nameof(value));
            }
            WeightsArray = array;
            ComputeCumulativeProbabilities();
        }
    }

    /// <inheritdoc/>
    public bool Replacement
    {
        get => ReplacementEnabled;
        set => ReplacementEnabled = value;
    }

    /// <inheritdoc/>
    public int? NumSamples
    {
        get => NumSamplesOverride;
        set => NumSamplesOverride = value;
    }

    /// <summary>
    /// Computes cumulative probability distribution from weights.
    /// Override this method to customize probability computation in derived classes.
    /// </summary>
    protected virtual void ComputeCumulativeProbabilities()
    {
        ComputeCumulativeProbabilitiesCore();
    }

    /// <summary>
    /// Core implementation of cumulative probability computation.
    /// This method is called directly from the constructor to avoid virtual call issues.
    /// </summary>
    private void ComputeCumulativeProbabilitiesCore()
    {
        CumulativeProbabilities = new Tensor<double>(new[] { WeightsArray.Length });
        if (WeightsArray.Length == 0)
        {
            throw new ArgumentException("Total weight must be greater than zero.");
        }

        // Validate the weights with engine ops. The distribution is built in double (one cast), as the scalar
        // accumulation was: in T the running sum drifts for large datasets and saturates in half precision.
        var engine = AiDotNetEngine.Current;
        var weights = new Tensor<T>((T[])WeightsArray.Clone(), new[] { WeightsArray.Length }).Cast<double>();
        var negative = engine.TensorLessThan(weights, 0.0);
        if (engine.TensorMaxValue(negative) > 0.0)
        {
            // ArgMax returns the first maximum, i.e. the first negative weight.
            int i = engine.TensorArgMax(negative, 0).AsSpan()[0];
            T w = WeightsArray[i];
            throw new ArgumentException($"Weight at index {i} is negative ({w}). Weights must be non-negative.");
        }

        // Running sum (left to right, like the previous scalar accumulation); its last entry is the total
        var runningSum = engine.TensorCumSum(weights, 0);
        double sum = runningSum.AsSpan()[WeightsArray.Length - 1];

        if (sum <= 0.0)
        {
            throw new ArgumentException("Total weight must be greater than zero.");
        }

        // Compute cumulative probabilities: cumulative sum of w / sum
        CumulativeProbabilities = engine.TensorCumSum(engine.TensorDivideScalar(weights, sum), 0);

        // Ensure last element is exactly 1.0 to avoid floating point issues
        CumulativeProbabilities[WeightsArray.Length - 1] = 1.0;
    }

    /// <summary>
    /// Samples an index based on the cumulative probabilities using binary search.
    /// </summary>
    /// <returns>A randomly selected index based on weights.</returns>
    protected int SampleWeightedIndex()
    {
        // First entry with r <= cumulative, capped at the last index like the former binary search.
        return CumulativeSearch.FirstAtOrAbove<double>(CumulativeProbabilities.AsSpan(), Random.NextDouble());
    }
}

/// <summary>
/// Binary search over a non-decreasing cumulative distribution for a single draw.
/// </summary>
/// <remarks>
/// A sampler draws one index at a time, so each draw must stay O(log n). The engine's search-sorted is
/// built for batches of values: for a single value it allocates two tensors, and on <see cref="float"/> it
/// first scans the whole sequence for NaN, which made every draw O(n).
/// </remarks>
internal static class CumulativeSearch
{
    /// <summary>
    /// Returns the first index whose cumulative value is not below <paramref name="value"/>, or the last
    /// index when every entry is below it.
    /// </summary>
    internal static int FirstAtOrAbove<T>(ReadOnlySpan<T> cumulative, T value)
    {
        var comparer = Comparer<T>.Default;
        int left = 0;
        int right = cumulative.Length - 1;

        while (left < right)
        {
            int mid = left + ((right - left) >> 1);
            if (comparer.Compare(cumulative[mid], value) < 0)
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
}

/// <summary>
/// Base class for epoch-adaptive samplers that change behavior over training epochs.
/// </summary>
/// <typeparam name="T">The numeric type for scores/values.</typeparam>
/// <remarks>
/// <para>
/// EpochAdaptiveSamplerBase is for samplers like curriculum learning and self-paced
/// learning that adjust their sampling strategy based on the current epoch.
/// </para>
/// </remarks>
public abstract class EpochAdaptiveSamplerBase<T> : DataSamplerBase
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The total number of epochs for curriculum progression.
    /// </summary>
    protected int TotalEpochs;

    /// <summary>
    /// Initializes a new instance of the EpochAdaptiveSamplerBase class.
    /// </summary>
    /// <param name="totalEpochs">Total number of epochs for progression.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected EpochAdaptiveSamplerBase(int totalEpochs, int? seed = null)
        : base(seed)
    {
        TotalEpochs = totalEpochs > 0 ? totalEpochs : throw new ArgumentOutOfRangeException(nameof(totalEpochs));
    }

    /// <summary>
    /// Gets the current progress through the curriculum (0.0 to 1.0).
    /// </summary>
    protected double Progress => Math.Min(1.0, (double)CurrentEpoch / TotalEpochs);

    /// <inheritdoc/>
    public override void OnEpochStart(int epoch)
    {
        base.OnEpochStart(epoch);
        OnEpochStartCore(epoch);
    }

    /// <summary>
    /// Called when a new epoch starts. Override to implement epoch-specific behavior.
    /// </summary>
    /// <param name="epoch">The current epoch number (0-based).</param>
    protected virtual void OnEpochStartCore(int epoch)
    {
        // Default implementation does nothing - override in derived classes
    }
}
