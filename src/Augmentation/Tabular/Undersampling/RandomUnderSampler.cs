using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular.Undersampling;

/// <summary>
/// Implements random undersampling to balance imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Random undersampling reduces the majority class by randomly
/// removing samples until the classes are balanced. This is the simplest undersampling method.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Count samples in majority and minority classes</item>
/// <item>Randomly select N samples from the majority class</item>
/// <item>N is determined by the target ratio</item>
/// </list>
/// </para>
///
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item>Simple and fast</item>
/// <item>Reduces training time by having fewer samples</item>
/// <item>May improve some classifiers by removing redundant samples</item>
/// </list>
/// </para>
///
/// <para><b>Disadvantages:</b>
/// <list type="bullet">
/// <item>May discard useful information from the majority class</item>
/// <item>Can lead to underfitting if too many samples are removed</item>
/// <item>Random selection doesn't consider sample importance</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When the majority class is very large and redundant</item>
/// <item>When training time is a concern</item>
/// <item>As a baseline before trying more sophisticated methods</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomUnderSampler<T> : IUnderSampler<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets the target ratio between minority and majority samples after undersampling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A ratio of 1.0 means equal numbers of minority and majority
    /// samples. A ratio of 0.5 means twice as many majority samples as minority.</para>
    /// <para>Default: 1.0 (balanced classes)</para>
    /// </remarks>
    public double SamplingRatio { get; }

    /// <summary>
    /// Gets whether to use replacement when sampling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Without replacement (default), each sample can only be
    /// selected once. With replacement, the same sample can appear multiple times.</para>
    /// <para>Default: false</para>
    /// </remarks>
    public bool WithReplacement { get; }

    /// <summary>
    /// Gets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Setting a seed ensures the same samples are selected each time.</para>
    /// </remarks>
    public int? RandomSeed { get; }

    /// <summary>
    /// Creates a new random undersampler.
    /// </summary>
    /// <param name="samplingRatio">Target ratio of minority to majority (default: 1.0 for balance).</param>
    /// <param name="withReplacement">Allow sampling the same instance multiple times (default: false).</param>
    /// <param name="randomSeed">Random seed for reproducibility (optional).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use default parameters for full balancing. Decrease
    /// samplingRatio if you want to keep more majority samples.</para>
    /// </remarks>
    public RandomUnderSampler(
        double samplingRatio = 1.0,
        bool withReplacement = false,
        int? randomSeed = null)
    {
        if (samplingRatio <= 0 || samplingRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingRatio),
                "Sampling ratio must be in range (0, 1].");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        SamplingRatio = samplingRatio;
        WithReplacement = withReplacement;
        RandomSeed = randomSeed;
    }

    /// <summary>
    /// Performs random undersampling on the majority class.
    /// </summary>
    /// <param name="data">The full dataset.</param>
    /// <param name="labels">Class labels for each sample.</param>
    /// <param name="minorityLabel">The label value for the minority class.</param>
    /// <returns>Tuple of (undersampled data, undersampled labels).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method identifies the majority class and randomly
    /// removes samples to achieve the target balance with the minority class.</para>
    /// </remarks>
    public (Matrix<T> Data, Vector<T> Labels) Undersample(
        Matrix<T> data,
        Vector<T> labels,
        T minorityLabel)
    {
        int rows = data.Rows;
        int cols = data.Columns;

        // Separate minority and majority indices
        var minorityIndices = new List<int>();
        var majorityIndices = new List<int>();

        double minorityLabelVal = _numOps.ToDouble(minorityLabel);

        for (int i = 0; i < rows; i++)
        {
            if (_numOps.ToDouble(labels[i]).Equals(minorityLabelVal))
            {
                minorityIndices.Add(i);
            }
            else
            {
                majorityIndices.Add(i);
            }
        }

        // Calculate target number of majority samples
        int targetMajority = (int)Math.Ceiling(minorityIndices.Count / SamplingRatio);

        // If we already have fewer majority samples, no undersampling needed
        if (majorityIndices.Count <= targetMajority)
        {
            return (data.Clone(), labels.Clone());
        }

        // Randomly select majority samples
        var rand = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        var selectedMajorityIndices = SelectRandomIndices(
            majorityIndices, targetMajority, WithReplacement, rand);

        // Combine selected majority with all minority
        int totalSamples = minorityIndices.Count + selectedMajorityIndices.Count;
        var resultData = new Matrix<T>(totalSamples, cols);
        var resultLabels = new Vector<T>(totalSamples);

        int idx = 0;

        // Add all minority samples
        foreach (int minIdx in minorityIndices)
        {
            for (int c = 0; c < cols; c++)
            {
                resultData[idx, c] = data[minIdx, c];
            }
            resultLabels[idx] = labels[minIdx];
            idx++;
        }

        // Add selected majority samples
        foreach (int majIdx in selectedMajorityIndices)
        {
            for (int c = 0; c < cols; c++)
            {
                resultData[idx, c] = data[majIdx, c];
            }
            resultLabels[idx] = labels[majIdx];
            idx++;
        }

        return (resultData, resultLabels);
    }

    /// <summary>
    /// Randomly selects indices from the given list.
    /// </summary>
    /// <param name="indices">The available indices.</param>
    /// <param name="count">Number to select.</param>
    /// <param name="withReplacement">Whether to allow duplicates.</param>
    /// <param name="rand">Random number generator.</param>
    /// <returns>List of selected indices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This implements the random selection with or without
    /// replacement. Without replacement means each index can only appear once.</para>
    /// </remarks>
    private List<int> SelectRandomIndices(List<int> indices, int count, bool withReplacement, Random rand)
    {
        var selected = new List<int>();

        if (withReplacement)
        {
            for (int i = 0; i < count; i++)
            {
                selected.Add(indices[rand.Next(indices.Count)]);
            }
        }
        else
        {
            // Fisher-Yates shuffle and take first 'count' elements
            var shuffled = new List<int>(indices);
            for (int i = shuffled.Count - 1; i > 0; i--)
            {
                int j = rand.Next(i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }
            selected = shuffled.Take(count).ToList();
        }

        return selected;
    }
}
