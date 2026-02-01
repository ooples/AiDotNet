using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Leave-One-Out Cross-Validation (LOOCV): each sample is used once as validation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> LOOCV is the extreme case of K-Fold where K equals the number of samples:
/// <list type="bullet">
/// <item>Each sample gets a turn as the single validation point</item>
/// <item>Maximizes training data usage (N-1 samples for training)</item>
/// <item>Provides nearly unbiased estimate of model performance</item>
/// <item>Computationally expensive: requires N model trainings</item>
/// </list>
/// </para>
/// <para>
/// <b>When to use:</b>
/// <list type="bullet">
/// <item>Very small datasets where you can't afford to hold out much data</item>
/// <item>When computational cost is not a concern</item>
/// <item>When you need the most accurate estimate possible</item>
/// </list>
/// </para>
/// <para>
/// <b>Caution:</b> For datasets with N > 1000, consider using K-Fold instead.
/// LOOCV can also have high variance in some scenarios.
/// </para>
/// </remarks>
public class LeaveOneOutStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private int _dataSize;

    public string Name => "LeaveOneOut";
    public int NumSplits => _dataSize > 0 ? _dataSize : -1; // -1 indicates "depends on data"
    public string Description => "Leave-one-out cross-validation - each sample is held out once.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize < 2)
            throw new ArgumentException("Need at least 2 samples for leave-one-out.", nameof(dataSize));

        _dataSize = dataSize;

        for (int i = 0; i < dataSize; i++)
        {
            // Validation is just sample i
            var validationIndices = new int[] { i };

            // Training is all samples except i
            var trainIndices = new int[dataSize - 1];
            int trainIdx = 0;
            for (int j = 0; j < dataSize; j++)
            {
                if (j != i) trainIndices[trainIdx++] = j;
            }

            yield return (trainIndices, validationIndices);
        }
    }
}
