using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// Leave-One-Out cross-validation where each sample is the test set once.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Leave-One-Out (LOO) is the extreme version of K-Fold where k = n (number of samples).
/// Each sample is tested individually while all other samples are used for training.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// <code>
/// Sample 1: [Test][Train][Train][Train][Train]...
/// Sample 2: [Train][Test][Train][Train][Train]...
/// Sample 3: [Train][Train][Test][Train][Train]...
/// ...
/// </code>
/// </para>
/// <para>
/// <b>Pros:</b>
/// - Uses maximum training data (n-1 samples)
/// - Every sample gets tested
/// - No randomness - results are deterministic
/// </para>
/// <para>
/// <b>Cons:</b>
/// - Very slow: requires n model fits
/// - High variance in estimates
/// - Only practical for small datasets (&lt;100 samples)
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Very small datasets where every sample matters
/// - When you need deterministic (non-random) evaluation
/// - Medical/scientific studies with limited data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LeaveOneOutSplitter<T> : DataSplitterBase<T>
{
    private int _nSamples;

    /// <summary>
    /// Creates a new Leave-One-Out cross-validation splitter.
    /// </summary>
    /// <remarks>
    /// No parameters needed - LOO always uses each sample as test once.
    /// Shuffle parameter doesn't apply since we iterate through all samples.
    /// </remarks>
    public LeaveOneOutSplitter() : base(shuffle: false, randomSeed: 42)
    {
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSamples;

    /// <inheritdoc/>
    public override string Description => "Leave-One-Out cross-validation";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        _nSamples = X.Rows;

        if (_nSamples > 1000)
        {
            // Warning but don't throw - user may know what they're doing
            Console.WriteLine(
                $"Warning: LOO with {_nSamples} samples will require {_nSamples} model fits. " +
                "Consider using K-Fold for large datasets.");
        }

        for (int i = 0; i < _nSamples; i++)
        {
            // Test set is just sample i
            var testIndices = new[] { i };

            // Train set is everything except sample i
            var trainIndices = new int[_nSamples - 1];
            int trainIdx = 0;
            for (int j = 0; j < _nSamples; j++)
            {
                if (j != i)
                {
                    trainIndices[trainIdx++] = j;
                }
            }

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: i, totalFolds: _nSamples);
        }
    }
}
