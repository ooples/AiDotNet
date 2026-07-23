using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Blocked K-Fold: K-Fold with temporal blocking (gap) between train and validation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Blocked K-Fold adds a gap between training and validation:
/// <list type="bullet">
/// <item>Prevents data leakage from temporal correlation</item>
/// <item>Important for time-series-like data in standard K-Fold</item>
/// <item>The "gap" samples are excluded from both train and validation</item>
/// </list>
/// </para>
/// <para><b>Use case:</b> When data has temporal ordering but you want K-Fold style validation:
/// <list type="bullet">
/// <item>Financial features computed from rolling windows</item>
/// <item>User behavior data with session effects</item>
/// <item>Any data where adjacent samples are correlated</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BlockedKFoldStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _nFolds;
    private readonly int _gapSize;
    private readonly bool _shuffle;
    private readonly int? _randomSeed;

    public string Name => $"Blocked {_nFolds}-Fold (gap={_gapSize})";
    public string Description => "K-Fold with gap between train and validation to prevent leakage.";
    public int NumSplits => _nFolds;

    /// <summary>
    /// Initializes Blocked K-Fold.
    /// </summary>
    /// <param name="nFolds">Number of folds. Default: 5.</param>
    /// <param name="gapSize">Number of samples to exclude around validation. Default: 1.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default: false (preserve order).</param>
    /// <param name="randomSeed">Random seed for shuffling.</param>
    public BlockedKFoldStrategy(int nFolds = 5, int gapSize = 1, bool shuffle = false, int? randomSeed = null)
    {
        if (nFolds < 2)
            throw new ArgumentException("Number of folds must be at least 2.");
        if (gapSize < 0)
            throw new ArgumentException("Gap size cannot be negative.");

        _nFolds = nFolds;
        _gapSize = gapSize;
        _shuffle = shuffle;
        _randomSeed = randomSeed;
    }

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize <= 0)
            throw new ArgumentException("Data size must be positive.", nameof(dataSize));
        if (dataSize < _nFolds)
            throw new ArgumentException($"Cannot have {_nFolds} folds with only {dataSize} samples.", nameof(dataSize));

        var indices = Enumerable.Range(0, dataSize).ToArray();

        if (_shuffle)
        {
            var random = _randomSeed.HasValue
                ? RandomHelper.CreateSeededRandom(_randomSeed.Value)
                : RandomHelper.CreateSecureRandom();
            indices = indices.OrderBy(_ => random.Next()).ToArray();
        }

        int foldSize = dataSize / _nFolds;

        for (int fold = 0; fold < _nFolds; fold++)
        {
            int valStart = fold * foldSize;
            int valEnd = (fold == _nFolds - 1) ? dataSize : (fold + 1) * foldSize;

            // Define exclusion zone (validation + gap on both sides)
            int excludeStart = Math.Max(0, valStart - _gapSize);
            int excludeEnd = Math.Min(dataSize, valEnd + _gapSize);

            var trainIndices = new List<int>();
            var valIndices = new List<int>();

            for (int i = 0; i < dataSize; i++)
            {
                if (i >= valStart && i < valEnd)
                {
                    valIndices.Add(indices[i]);
                }
                else if (i < excludeStart || i >= excludeEnd)
                {
                    trainIndices.Add(indices[i]);
                }
                // Gap samples (excludeStart <= i < valStart || valEnd <= i < excludeEnd) are excluded
            }

            yield return (trainIndices.ToArray(), valIndices.ToArray());
        }
    }
}
