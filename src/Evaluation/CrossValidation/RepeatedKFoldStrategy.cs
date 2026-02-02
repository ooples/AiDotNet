using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Repeated K-Fold: runs K-Fold multiple times with different random shuffles.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Repeated K-Fold reduces variance by averaging over multiple K-Fold runs:
/// <list type="bullet">
/// <item>Runs standard K-Fold multiple times (e.g., 10 times)</item>
/// <item>Each repetition uses a different random shuffle</item>
/// <item>Provides more stable estimates than single K-Fold</item>
/// <item>Total splits = K × Repetitions (e.g., 5-fold × 10 reps = 50 evaluations)</item>
/// </list>
/// </para>
/// <para>
/// <b>Common configurations:</b>
/// <list type="bullet">
/// <item>5-fold × 2 repetitions (10 evaluations) - quick estimate</item>
/// <item>10-fold × 10 repetitions (100 evaluations) - robust estimate</item>
/// <item>5-fold × 10 repetitions (50 evaluations) - balanced choice</item>
/// </list>
/// </para>
/// </remarks>
public class RepeatedKFoldStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _k;
    private readonly int _repetitions;
    private readonly int? _randomSeed;

    /// <summary>
    /// Initializes Repeated K-Fold cross-validation.
    /// </summary>
    /// <param name="k">Number of folds per repetition. Default is 5.</param>
    /// <param name="repetitions">Number of times to repeat K-Fold. Default is 10.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public RepeatedKFoldStrategy(int k = 5, int repetitions = 10, int? randomSeed = null)
    {
        if (k < 2) throw new ArgumentException("K must be at least 2.", nameof(k));
        if (repetitions < 1) throw new ArgumentException("Repetitions must be at least 1.", nameof(repetitions));

        _k = k;
        _repetitions = repetitions;
        _randomSeed = randomSeed;
    }

    public string Name => $"Repeated {_k}-Fold (×{_repetitions})";
    public int NumSplits => _k * _repetitions;
    public string Description => $"{_k}-fold cross-validation repeated {_repetitions} times.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        // Convert span to array before yield operations (spans can't be captured across yield boundaries)
        return SplitInternal(dataSize, labels.IsEmpty ? null : labels.ToArray());
    }

    private IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> SplitInternal(int dataSize, T[]? labelsArray)
    {
        if (dataSize < _k)
            throw new ArgumentException($"Cannot have {_k} folds with only {dataSize} samples.", nameof(dataSize));

        var baseRandom = _randomSeed.HasValue ? RandomHelper.CreateSeededRandom(_randomSeed.Value) : RandomHelper.CreateSecureRandom();

        for (int rep = 0; rep < _repetitions; rep++)
        {
            // Use a different seed for each repetition
            int repSeed = baseRandom.Next();
            var kfold = new KFoldStrategy<T>(_k, shuffle: true, randomSeed: repSeed);

            foreach (var split in kfold.Split(dataSize, labelsArray))
            {
                yield return split;
            }
        }
    }
}
