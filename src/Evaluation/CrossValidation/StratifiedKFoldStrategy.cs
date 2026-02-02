using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Stratified K-Fold: K-Fold that preserves the percentage of samples for each class.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Stratified K-Fold is essential for classification problems, especially
/// with imbalanced classes:
/// <list type="bullet">
/// <item>Each fold has approximately the same class distribution as the full dataset</item>
/// <item>Prevents folds where a rare class is entirely missing</item>
/// <item>Produces more reliable estimates for imbalanced datasets</item>
/// </list>
/// </para>
/// <para>
/// <b>Example:</b> If your data has 70% class A and 30% class B, each fold will have
/// approximately 70% A and 30% B (not 100% A or 100% B which could happen with regular K-Fold).
/// </para>
/// </remarks>
public class StratifiedKFoldStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _k;
    private readonly bool _shuffle;
    private readonly int? _randomSeed;

    /// <summary>
    /// Initializes Stratified K-Fold cross-validation.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="shuffle">Whether to shuffle data within each class before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public StratifiedKFoldStrategy(int k = 5, bool shuffle = true, int? randomSeed = null)
    {
        if (k < 2) throw new ArgumentException("K must be at least 2.", nameof(k));
        _k = k;
        _shuffle = shuffle;
        _randomSeed = randomSeed;
    }

    public string Name => $"Stratified {_k}-Fold";
    public int NumSplits => _k;
    public string Description => $"Stratified {_k}-fold cross-validation preserving class distribution.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        // Convert span to array before yield operations (spans can't be captured across yield boundaries)
        return SplitInternal(dataSize, labels.IsEmpty ? null : labels.ToArray());
    }

    private IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> SplitInternal(int dataSize, T[]? labelsArray)
    {
        if (labelsArray == null || labelsArray.Length == 0)
            throw new ArgumentException("Stratified K-Fold requires labels.", nameof(labelsArray));
        if (labelsArray.Length != dataSize)
            throw new ArgumentException("Labels must have same length as data.", nameof(labelsArray));
        if (dataSize < _k)
            throw new ArgumentException($"Cannot have {_k} folds with only {dataSize} samples.", nameof(dataSize));

        var random = _randomSeed.HasValue ? RandomHelper.CreateSeededRandom(_randomSeed.Value) : new Random();

        // Group indices by class using tolerance-based equality for floating-point labels
        // This avoids merging distinct classes that happen to round to the same integer
        var classSamples = new Dictionary<double, List<int>>();
        const double tolerance = 1e-10;

        for (int i = 0; i < dataSize; i++)
        {
            double labelValue = NumOps.ToDouble(labelsArray[i]);

            // Find existing key within tolerance, or use the label value as a new key
            double? existingKey = null;
            foreach (var key in classSamples.Keys)
            {
                if (Math.Abs(key - labelValue) < tolerance)
                {
                    existingKey = key;
                    break;
                }
            }

            double classLabel = existingKey ?? labelValue;
            if (!classSamples.ContainsKey(classLabel))
                classSamples[classLabel] = new List<int>();
            classSamples[classLabel].Add(i);
        }

        // Shuffle within each class if requested
        if (_shuffle)
        {
            foreach (var samples in classSamples.Values)
            {
                for (int i = samples.Count - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (samples[i], samples[j]) = (samples[j], samples[i]);
                }
            }
        }

        // Create fold assignments for each class
        var foldAssignments = new int[dataSize];
        foreach (var kvp in classSamples)
        {
            var samples = kvp.Value;
            int n = samples.Count;
            int baseFoldSize = n / _k;
            int remainder = n % _k;

            int sampleIdx = 0;
            for (int fold = 0; fold < _k; fold++)
            {
                int foldSize = baseFoldSize + (fold < remainder ? 1 : 0);
                for (int i = 0; i < foldSize && sampleIdx < n; i++)
                {
                    foldAssignments[samples[sampleIdx++]] = fold;
                }
            }
        }

        // Generate train/validation splits for each fold
        for (int fold = 0; fold < _k; fold++)
        {
            var validationIndices = new List<int>();
            var trainIndices = new List<int>();

            for (int i = 0; i < dataSize; i++)
            {
                if (foldAssignments[i] == fold)
                    validationIndices.Add(i);
                else
                    trainIndices.Add(i);
            }

            yield return (trainIndices.ToArray(), validationIndices.ToArray());
        }
    }
}
