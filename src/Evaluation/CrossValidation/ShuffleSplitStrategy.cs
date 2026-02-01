using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Shuffle Split (Monte Carlo Cross-Validation): random train/test splits repeated multiple times.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Shuffle Split randomly samples training and test sets:
/// <list type="bullet">
/// <item>Unlike K-Fold, each split is independent (some samples may appear multiple times in test)</item>
/// <item>You control exact train/test sizes</item>
/// <item>More flexible than K-Fold - can have any test proportion</item>
/// <item>Good for very large datasets where you want quick estimates</item>
/// </list>
/// </para>
/// <para>
/// <b>Comparison with K-Fold:</b>
/// <list type="bullet">
/// <item>K-Fold: Each sample appears in test exactly once</item>
/// <item>Shuffle Split: Samples may appear in test 0, 1, or multiple times</item>
/// </list>
/// </para>
/// </remarks>
public class ShuffleSplitStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numSplits;
    private readonly double _testSize;
    private readonly int? _randomSeed;

    /// <summary>
    /// Initializes Shuffle Split cross-validation.
    /// </summary>
    /// <param name="numSplits">Number of random splits to generate. Default is 10.</param>
    /// <param name="testSize">Proportion of data for test set (0.0-1.0). Default is 0.2 (20%).</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public ShuffleSplitStrategy(int numSplits = 10, double testSize = 0.2, int? randomSeed = null)
    {
        if (numSplits < 1) throw new ArgumentException("Number of splits must be at least 1.", nameof(numSplits));
        if (testSize <= 0 || testSize >= 1) throw new ArgumentException("Test size must be between 0 and 1.", nameof(testSize));

        _numSplits = numSplits;
        _testSize = testSize;
        _randomSeed = randomSeed;
    }

    public string Name => "ShuffleSplit";
    public int NumSplits => _numSplits;
    public string Description => $"Random {(int)(_testSize * 100)}% test splits, repeated {_numSplits} times.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        int testCount = Math.Max(1, (int)(dataSize * _testSize));
        int trainCount = dataSize - testCount;

        if (trainCount < 1)
            throw new ArgumentException("Train size would be less than 1.", nameof(dataSize));

        var random = _randomSeed.HasValue ? RandomHelper.CreateSeededRandom(_randomSeed.Value) : new Random();

        for (int split = 0; split < _numSplits; split++)
        {
            // Create and shuffle indices
            var indices = new int[dataSize];
            for (int i = 0; i < dataSize; i++) indices[i] = i;

            // Fisher-Yates shuffle
            for (int i = dataSize - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Split into train and test
            var trainIndices = new int[trainCount];
            var testIndices = new int[testCount];

            Array.Copy(indices, 0, trainIndices, 0, trainCount);
            Array.Copy(indices, trainCount, testIndices, 0, testCount);

            yield return (trainIndices, testIndices);
        }
    }
}
