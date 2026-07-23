using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Leave-P-Out cross-validation: train on N-P samples, validate on P samples for all combinations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Leave-P-Out is a generalization of Leave-One-Out:
/// <list type="bullet">
/// <item>P=1: Leave-One-Out (LOO)</item>
/// <item>P=2: Leave-Two-Out (exhaustive but expensive)</item>
/// <item>Number of folds = C(N,P) = N!/(P!(N-P)!)</item>
/// <item>Very thorough but computationally expensive for large datasets</item>
/// </list>
/// </para>
/// <para><b>Warning:</b> The number of combinations grows rapidly. For N=20 and P=2, there are 190 folds.
/// For N=100 and P=2, there are 4,950 folds. Use with caution on larger datasets.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LeavePOutStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _p;
    private readonly int? _maxFolds;

    public string Name => $"Leave-{_p}-Out";
    public string Description => $"Leave-{_p}-Out cross-validation (exhaustive for small datasets).";
    public int NumSplits { get; private set; }

    /// <summary>
    /// Initializes Leave-P-Out cross-validation.
    /// </summary>
    /// <param name="p">Number of samples to leave out in each fold. Default: 2.</param>
    /// <param name="maxFolds">Maximum number of folds to generate (for computational limits). Default: 1000.</param>
    public LeavePOutStrategy(int p = 2, int? maxFolds = 1000)
    {
        if (p < 1)
            throw new ArgumentException("P must be at least 1.");
        _p = p;
        _maxFolds = maxFolds;
    }

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize <= _p)
            throw new ArgumentException($"Dataset size ({dataSize}) must be greater than P ({_p}).");

        // Use iterator for memory efficiency, but convert to list for combination generation
        return SplitInternal(dataSize);
    }

    private IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> SplitInternal(int dataSize)
    {
        var allIndices = Enumerable.Range(0, dataSize).ToArray();
        int foldCount = 0;

        foreach (var validationIndices in GetCombinations(allIndices, _p))
        {
            if (_maxFolds.HasValue && foldCount >= _maxFolds.Value)
                yield break;

            var valSet = new HashSet<int>(validationIndices);
            var trainIndices = allIndices.Where(i => !valSet.Contains(i)).ToArray();

            foldCount++;
            yield return (trainIndices, validationIndices);
        }

        NumSplits = foldCount;
    }

    private static IEnumerable<int[]> GetCombinations(int[] elements, int k)
    {
        int n = elements.Length;
        if (k > n) yield break;

        var indices = new int[k];
        for (int i = 0; i < k; i++)
            indices[i] = i;

        while (true)
        {
            yield return indices.Select(i => elements[i]).ToArray();

            // Find rightmost element that can be incremented
            int pos = k - 1;
            while (pos >= 0 && indices[pos] == n - k + pos)
                pos--;

            if (pos < 0) yield break;

            indices[pos]++;
            for (int i = pos + 1; i < k; i++)
                indices[i] = indices[i - 1] + 1;
        }
    }
}
