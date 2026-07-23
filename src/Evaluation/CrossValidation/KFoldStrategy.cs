using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// K-Fold cross-validation: splits data into K equal-sized folds, using each as validation once.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> K-Fold is the most common cross-validation approach:
/// <list type="bullet">
/// <item>Data is divided into K equal parts (folds)</item>
/// <item>Model is trained K times, each time using a different fold for validation</item>
/// <item>Results are averaged across all K runs</item>
/// </list>
/// Common choices: K=5 or K=10. Higher K means more compute but lower variance in estimates.</para>
/// <para>
/// <b>Example with K=5:</b>
/// <code>
/// Fold 1: Train on folds 2,3,4,5 | Validate on fold 1
/// Fold 2: Train on folds 1,3,4,5 | Validate on fold 2
/// Fold 3: Train on folds 1,2,4,5 | Validate on fold 3
/// Fold 4: Train on folds 1,2,3,5 | Validate on fold 4
/// Fold 5: Train on folds 1,2,3,4 | Validate on fold 5
/// </code>
/// </para>
/// </remarks>
public class KFoldStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _k;
    private readonly bool _shuffle;
    private readonly int? _randomSeed;

    /// <summary>
    /// Initializes K-Fold cross-validation.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="shuffle">Whether to shuffle data before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public KFoldStrategy(int k = 5, bool shuffle = true, int? randomSeed = null)
    {
        if (k < 2) throw new ArgumentException("K must be at least 2.", nameof(k));
        _k = k;
        _shuffle = shuffle;
        _randomSeed = randomSeed;
    }

    public string Name => $"{_k}-Fold";
    public int NumSplits => _k;
    public string Description => $"Standard {_k}-fold cross-validation{(_shuffle ? " with shuffling" : "")}.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize < _k)
            throw new ArgumentException($"Cannot have {_k} folds with only {dataSize} samples.", nameof(dataSize));

        // Create indices array
        var indices = new int[dataSize];
        for (int i = 0; i < dataSize; i++) indices[i] = i;

        // Shuffle if requested
        if (_shuffle)
        {
            var random = _randomSeed.HasValue ? RandomHelper.CreateSeededRandom(_randomSeed.Value) : RandomHelper.CreateSecureRandom();
            for (int i = dataSize - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        // Calculate fold sizes
        int baseFoldSize = dataSize / _k;
        int remainder = dataSize % _k;

        int startIdx = 0;
        for (int fold = 0; fold < _k; fold++)
        {
            // Some folds get one extra sample to distribute the remainder
            int foldSize = baseFoldSize + (fold < remainder ? 1 : 0);
            int endIdx = startIdx + foldSize;

            // Validation indices are this fold
            var validationIndices = new int[foldSize];
            Array.Copy(indices, startIdx, validationIndices, 0, foldSize);

            // Train indices are all other folds
            var trainIndices = new int[dataSize - foldSize];
            int trainIdx = 0;
            for (int i = 0; i < startIdx; i++) trainIndices[trainIdx++] = indices[i];
            for (int i = endIdx; i < dataSize; i++) trainIndices[trainIdx++] = indices[i];

            yield return (trainIndices, validationIndices);
            startIdx = endIdx;
        }
    }
}
