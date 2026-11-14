using AiDotNet.Interfaces;
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
public class WeightedSampler<T> : WeightedSamplerBase<T>
{
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
        : base(weights, numSamples, replacement, seed)
    {
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        int numToSample = NumSamplesOverride ?? WeightsArray.Length;

        if (ReplacementEnabled)
        {
            // Sampling with replacement using inverse transform sampling
            for (int i = 0; i < numToSample; i++)
            {
                yield return SampleWeightedIndex();
            }
        }
        else
        {
            // Sampling without replacement
            // For efficiency, we track which indices have been selected
            var availableIndices = Enumerable.Range(0, WeightsArray.Length).ToList();
            var engine = AiDotNetEngine.Current;
            var allWeights = new Tensor<T>((T[])WeightsArray.Clone(), new[] { WeightsArray.Length });

            for (int i = 0; i < Math.Min(numToSample, WeightsArray.Length); i++)
            {
                // Recompute probabilities excluding selected indices: gather the remaining weights and take
                // their left-to-right running sum (its last entry is the total).
                var available = engine.TensorIndexSelect(
                    allWeights, new Tensor<int>(availableIndices.ToArray(), new[] { availableIndices.Count }), 0);
                T sum = engine.TensorCumSum(available, 0).AsSpan()[availableIndices.Count - 1];

                if (NumOps.LessThanOrEquals(sum, NumOps.Zero))
                {
                    yield break;
                }

                // Build cumulative distribution over available indices and pick the first entry with
                // u <= cumulative (left search-sorted); default to the last if none qualifies.
                T u = NumOps.FromDouble(Random.NextDouble());
                var cumulative = engine.TensorCumSum(engine.TensorDivideScalar(available, sum), 0);
                int firstAtOrAbove = engine.TensorSearchSorted(
                    cumulative, new Tensor<T>(new[] { u }, new[] { 1 }), right: false).AsSpan()[0];
                int selectedLocalIndex = Math.Min(firstAtOrAbove, availableIndices.Count - 1);

                int selectedIndex = availableIndices[selectedLocalIndex];
                yield return selectedIndex;

                // Remove selected index
                availableIndices.RemoveAt(selectedLocalIndex);
            }
        }
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
        // Count samples per class - filter to valid labels explicitly
        int[] classCounts = new int[numClasses];
        foreach (int label in labels.Where(l => l >= 0 && l < numClasses))
        {
            classCounts[label]++;
        }

        int totalSamples = labels.Count;

        // Assign weight to each sample based on its class
        // Labels outside [0, numClasses) point at an extra zero-weight slot; the per-sample weights are
        // one engine gather from the class-weight table followed by one cast to T.
        if (labels.Count == 0)
        {
            return Array.Empty<T>();
        }

        // Weight per class (inverse of frequency): total / (numClasses * count) in T, zero for empty
        // classes and for the extra slot.
        var engine = AiDotNetEngine.Current;
        var countTable = new int[numClasses + 1];
        Array.Copy(classCounts, countTable, numClasses);
        var counts = new Tensor<int>(countTable, new[] { countTable.Length }).Cast<T>();
        var total = new Tensor<T>(new[] { countTable.Length });
        total.Fill(NumOps.FromDouble(totalSamples));
        var weightTable = engine.TensorWhere(
            engine.TensorGreaterThan(counts, NumOps.Zero),
            engine.TensorDivide(total, engine.TensorMultiplyScalar(counts, NumOps.FromDouble(numClasses))),
            new Tensor<T>(new[] { countTable.Length }));
        var slots = new int[labels.Count];
        for (int i = 0; i < labels.Count; i++)
        {
            int label = labels[i];
            slots[i] = (label >= 0 && label < numClasses) ? label : numClasses;
        }

        var gathered = engine.TensorIndexSelect(weightTable, new Tensor<int>(slots, new[] { slots.Length }), 0);
        return gathered.AsSpan().ToArray();
    }
}
