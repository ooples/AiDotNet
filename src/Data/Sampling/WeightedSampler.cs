using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// A sampler that samples indices based on their weights.
/// </summary>
/// <typeparam name="T">The numeric type for weights.</typeparam>
/// <remarks>
/// <para>
/// WeightedSampler selects samples with probability proportional to their weights.
/// This is useful for handling class imbalance, importance sampling, or focusing
/// training on harder examples.
/// </para>
/// <para><b>For Beginners:</b> This sampler lets you control how often each sample appears:
///
/// - Higher weight = more likely to be selected
/// - Lower weight = less likely to be selected
///
/// Common uses:
/// - **Class imbalance**: Give higher weights to underrepresented classes
/// - **Hard example mining**: Give higher weights to samples the model struggles with
/// - **Curriculum learning**: Adjust weights during training based on difficulty
///
/// Example:
/// <code>
/// // Dataset has 900 cats, 100 dogs
/// // Give dogs 9x higher weight to balance
/// var weights = labels.Select(l => l == "dog" ? 9.0f : 1.0f).ToList();
/// var sampler = new WeightedSampler&lt;float&gt;(weights);
/// </code>
/// </para>
/// </remarks>
public class WeightedSampler<T> : IWeightedSampler<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private T[] _weights;
    private double[] _cumulativeProbabilities;
    private bool _replacement;
    private int? _numSamples;
    private Random _random;

    /// <summary>
    /// Initializes a new instance of the WeightedSampler class.
    /// </summary>
    /// <param name="weights">The weight for each sample. Must be non-negative.</param>
    /// <param name="numSamples">Number of samples to draw per epoch. Defaults to dataset size.</param>
    /// <param name="replacement">Whether to sample with replacement.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <exception cref="ArgumentNullException">Thrown when weights is null.</exception>
    /// <exception cref="ArgumentException">Thrown when weights is empty or contains negative values.</exception>
    public WeightedSampler(
        IEnumerable<T> weights,
        int? numSamples = null,
        bool replacement = true,
        int? seed = null)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        _weights = weights.ToArray();
        if (_weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be empty.", nameof(weights));
        }

        _numSamples = numSamples;
        _replacement = replacement;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        _cumulativeProbabilities = new double[_weights.Length];
        ComputeCumulativeProbabilities();
    }

    /// <inheritdoc/>
    public int Length => _numSamples ?? _weights.Length;

    /// <inheritdoc/>
    public IReadOnlyList<T> Weights
    {
        get => _weights;
        set
        {
            _weights = value?.ToArray() ?? throw new ArgumentNullException(nameof(value));
            ComputeCumulativeProbabilities();
        }
    }

    /// <inheritdoc/>
    public bool Replacement
    {
        get => _replacement;
        set => _replacement = value;
    }

    /// <inheritdoc/>
    public int? NumSamples
    {
        get => _numSamples;
        set => _numSamples = value;
    }

    /// <summary>
    /// Computes cumulative probability distribution from weights.
    /// </summary>
    private void ComputeCumulativeProbabilities()
    {
        _cumulativeProbabilities = new double[_weights.Length];

        // Convert weights to doubles and compute sum
        double sum = 0;
        for (int i = 0; i < _weights.Length; i++)
        {
            double w = NumOps.ToDouble(_weights[i]);
            if (w < 0)
            {
                throw new ArgumentException($"Weight at index {i} is negative ({w}). Weights must be non-negative.");
            }
            sum += w;
        }

        if (sum <= 0)
        {
            throw new ArgumentException("Total weight must be greater than zero.");
        }

        // Compute cumulative probabilities
        double cumulative = 0;
        for (int i = 0; i < _weights.Length; i++)
        {
            cumulative += NumOps.ToDouble(_weights[i]) / sum;
            _cumulativeProbabilities[i] = cumulative;
        }

        // Ensure last element is exactly 1.0 to avoid floating point issues
        _cumulativeProbabilities[_weights.Length - 1] = 1.0;
    }

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        int numToSample = _numSamples ?? _weights.Length;

        if (_replacement)
        {
            // Sampling with replacement using inverse transform sampling
            for (int i = 0; i < numToSample; i++)
            {
                yield return SampleOne();
            }
        }
        else
        {
            // Sampling without replacement
            // For efficiency, we track which indices have been selected
            var selected = new HashSet<int>();
            var availableIndices = Enumerable.Range(0, _weights.Length).ToList();
            double[] currentProbabilities = new double[_weights.Length];

            for (int i = 0; i < Math.Min(numToSample, _weights.Length); i++)
            {
                // Recompute probabilities excluding selected indices
                double sum = 0;
                for (int j = 0; j < availableIndices.Count; j++)
                {
                    int idx = availableIndices[j];
                    sum += NumOps.ToDouble(_weights[idx]);
                }

                if (sum <= 0)
                {
                    yield break;
                }

                // Build cumulative distribution over available indices
                double cumulative = 0;
                int selectedLocalIndex = 0;
                double u = _random.NextDouble();

                for (int j = 0; j < availableIndices.Count; j++)
                {
                    int idx = availableIndices[j];
                    cumulative += NumOps.ToDouble(_weights[idx]) / sum;
                    if (u <= cumulative)
                    {
                        selectedLocalIndex = j;
                        break;
                    }
                }

                int selectedIndex = availableIndices[selectedLocalIndex];
                yield return selectedIndex;

                // Remove selected index
                availableIndices.RemoveAt(selectedLocalIndex);
            }
        }
    }

    /// <summary>
    /// Samples a single index using inverse transform sampling.
    /// </summary>
    private int SampleOne()
    {
        double u = _random.NextDouble();

        // Binary search for the index
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

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _random = RandomHelper.CreateSeededRandom(seed);
    }

    /// <summary>
    /// Creates weights that balance class frequencies.
    /// </summary>
    /// <param name="labels">The class labels for each sample.</param>
    /// <param name="numClasses">The number of classes.</param>
    /// <returns>Weights that inversely weight by class frequency.</returns>
    /// <remarks>
    /// <para>
    /// This helper method creates weights such that each class has equal total weight,
    /// effectively balancing an imbalanced dataset.
    /// </para>
    /// <para><b>For Beginners:</b> If you have 900 cats and 100 dogs, calling this method
    /// will give dogs 9x the weight of cats, so they get sampled equally often.
    /// </para>
    /// </remarks>
    public static T[] CreateBalancedWeights(IReadOnlyList<int> labels, int numClasses)
    {
        // Count samples per class
        int[] classCounts = new int[numClasses];
        foreach (int label in labels)
        {
            if (label >= 0 && label < numClasses)
            {
                classCounts[label]++;
            }
        }

        // Compute weight per class (inverse of frequency)
        double[] classWeights = new double[numClasses];
        int totalSamples = labels.Count;
        for (int c = 0; c < numClasses; c++)
        {
            if (classCounts[c] > 0)
            {
                classWeights[c] = (double)totalSamples / (numClasses * classCounts[c]);
            }
            else
            {
                classWeights[c] = 0;
            }
        }

        // Assign weight to each sample based on its class
        T[] weights = new T[labels.Count];
        for (int i = 0; i < labels.Count; i++)
        {
            int label = labels[i];
            if (label >= 0 && label < numClasses)
            {
                weights[i] = NumOps.FromDouble(classWeights[label]);
            }
            else
            {
                weights[i] = NumOps.Zero;
            }
        }

        return weights;
    }
}
