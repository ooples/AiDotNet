using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Bootstrap Cross-Validation: uses bootstrap sampling (sampling with replacement) for validation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Bootstrap creates training sets by sampling with replacement:
/// <list type="bullet">
/// <item>Each bootstrap sample is the same size as the original data</item>
/// <item>Some samples appear multiple times, others not at all</item>
/// <item>Out-of-bag (OOB) samples (not selected) form the validation set</item>
/// <item>On average, about 63.2% of samples appear in each bootstrap, 36.8% are OOB</item>
/// </list>
/// </para>
/// <para>
/// <b>Advantages:</b>
/// <list type="bullet">
/// <item>Can generate unlimited training sets</item>
/// <item>Works well with very small datasets</item>
/// <item>Provides good variance estimates</item>
/// </list>
/// </para>
/// <para>
/// <b>Variants:</b>
/// <list type="bullet">
/// <item>.632 Bootstrap: Weighted average of training and OOB error</item>
/// <item>.632+ Bootstrap: Adds correction for overfitting</item>
/// </list>
/// </para>
/// </remarks>
public class BootstrapStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numBootstraps;
    private readonly int? _randomSeed;

    /// <summary>
    /// Initializes Bootstrap cross-validation.
    /// </summary>
    /// <param name="numBootstraps">Number of bootstrap samples to generate. Default is 100.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public BootstrapStrategy(int numBootstraps = 100, int? randomSeed = null)
    {
        if (numBootstraps < 1) throw new ArgumentException("Number of bootstraps must be at least 1.", nameof(numBootstraps));
        _numBootstraps = numBootstraps;
        _randomSeed = randomSeed;
    }

    public string Name => "Bootstrap";
    public int NumSplits => _numBootstraps;
    public string Description => $"Bootstrap validation with {_numBootstraps} samples using out-of-bag evaluation.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize < 2)
            throw new ArgumentException("Need at least 2 samples for bootstrap.", nameof(dataSize));

        var random = _randomSeed.HasValue ? RandomHelper.CreateSeededRandom(_randomSeed.Value) : RandomHelper.CreateSecureRandom();

        int yielded = 0;
        int attempts = 0;
        int maxAttempts = _numBootstraps * 10; // Limit attempts to avoid infinite loop

        while (yielded < _numBootstraps && attempts < maxAttempts)
        {
            attempts++;

            // Sample with replacement for training
            var trainIndices = new int[dataSize];
            var selectedSet = new HashSet<int>();

            for (int i = 0; i < dataSize; i++)
            {
                int idx = random.Next(dataSize);
                trainIndices[i] = idx;
                selectedSet.Add(idx);
            }

            // Out-of-bag samples for validation
            var oobIndices = new List<int>();
            for (int i = 0; i < dataSize; i++)
            {
                if (!selectedSet.Contains(i))
                    oobIndices.Add(i);
            }

            // Skip if no OOB samples (rare but possible with small datasets)
            if (oobIndices.Count == 0) continue;

            yield return (trainIndices, oobIndices.ToArray());
            yielded++;
        }
    }
}
