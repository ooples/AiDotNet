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

            for (int i = 0; i < Math.Min(numToSample, WeightsArray.Length); i++)
            {
                // Recompute probabilities excluding selected indices
                double sum = 0;
                for (int j = 0; j < availableIndices.Count; j++)
                {
                    int idx = availableIndices[j];
                    sum += NumOps.ToDouble(WeightsArray[idx]);
                }

                if (sum <= 0)
                {
                    yield break;
                }

                // Build cumulative distribution over available indices
                double cumulative = 0;
                int selectedLocalIndex = availableIndices.Count - 1; // Default to last if none selected
                double u = Random.NextDouble();

                for (int j = 0; j < availableIndices.Count; j++)
                {
                    int idx = availableIndices[j];
                    cumulative += NumOps.ToDouble(WeightsArray[idx]) / sum;
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

        // Compute weight per class (inverse of frequency)
        double[] classWeights = new double[numClasses];
        int totalSamples = labels.Count;
        for (int c = 0; c < numClasses; c++)
        {
            // Cast to double before multiplication to prevent integer overflow
            classWeights[c] = classCounts[c] > 0
                ? (double)totalSamples / ((double)numClasses * classCounts[c])
                : 0;
        }

        // Assign weight to each sample based on its class
        T[] weights = new T[labels.Count];
        for (int i = 0; i < labels.Count; i++)
        {
            int label = labels[i];
            weights[i] = (label >= 0 && label < numClasses)
                ? NumOps.FromDouble(classWeights[label])
                : NumOps.Zero;
        }

        return weights;
    }
}
