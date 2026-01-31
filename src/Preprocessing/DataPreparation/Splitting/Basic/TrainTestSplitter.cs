using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Basic;

/// <summary>
/// Simple random train/test splitter that divides data into two sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the simplest and most common way to split your data.
/// It randomly divides your dataset into:
/// - Training set: Data your model learns from
/// - Test set: Data used to evaluate how well your model performs on unseen data
/// </para>
/// <para>
/// <b>Industry Standard:</b> An 80/20 split (80% train, 20% test) is very common.
/// For smaller datasets, you might use 70/30 to get more test samples.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Large datasets (10,000+ samples)
/// - Quick experiments
/// - When you don't need hyperparameter tuning (otherwise use train/val/test)
/// </para>
/// <para>
/// <b>Example Usage:</b>
/// <code>
/// var splitter = new TrainTestSplitter&lt;double&gt;(testSize: 0.2);
/// var result = splitter.Split(X, y);
/// // result.XTrain, result.yTrain - 80% of data
/// // result.XTest, result.yTest - 20% of data
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TrainTestSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;

    /// <summary>
    /// Creates a new train/test splitter.
    /// </summary>
    /// <param name="testSize">
    /// Proportion of data to use for testing (0.0 to 1.0).
    /// Default is 0.2 (20% test, 80% train) - the industry standard.
    /// </param>
    /// <param name="shuffle">Whether to shuffle data before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <exception cref="ArgumentOutOfRangeException">If testSize is not between 0 and 1.</exception>
    public TrainTestSplitter(double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize),
                "Test size must be between 0 and 1 (exclusive). Use 0.2 for 20% test.");
        }

        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override string Description => $"Train/Test split ({(1 - _testSize) * 100:F0}% train, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        var indices = GetShuffledIndices(nSamples);

        int testSize = Math.Max(1, (int)(nSamples * _testSize));
        int trainSize = nSamples - testSize;

        var trainIndices = indices.Take(trainSize).ToArray();
        var testIndices = indices.Skip(trainSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }

    /// <inheritdoc/>
    protected override (int[] TrainIndices, int[] TestIndices, int[]? ValidationIndices, int? FoldIndex, int? TotalFolds)
        SplitIndicesOnly(int nSamples, Vector<T>? y)
    {
        var indices = GetShuffledIndices(nSamples);

        int testSize = Math.Max(1, (int)(nSamples * _testSize));
        int trainSize = nSamples - testSize;

        var trainIndices = indices.Take(trainSize).ToArray();
        var testIndices = indices.Skip(trainSize).ToArray();

        return (trainIndices, testIndices, null, null, null);
    }
}
