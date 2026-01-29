using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Basic;

/// <summary>
/// Three-way splitter that divides data into training, validation, and test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This splitter creates three separate sets from your data:
/// - <b>Training set:</b> Data your model learns from (~70%)
/// - <b>Validation set:</b> Data used to tune hyperparameters and prevent overfitting (~15%)
/// - <b>Test set:</b> Data for final evaluation - never touch during training (~15%)
/// </para>
/// <para>
/// <b>Why Three Sets?</b>
/// If you only have train/test, you might tune your model to perform well on the test set,
/// which means you're "cheating" - your test set is no longer truly unseen data.
/// The validation set lets you tune without contaminating your final test evaluation.
/// </para>
/// <para>
/// <b>Industry Standard:</b> 70/15/15 or 60/20/20 splits are common.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Medium to large datasets
/// - When you need to tune hyperparameters
/// - Deep learning (where validation is essential for early stopping)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TrainValTestSplitter<T> : DataSplitterBase<T>
{
    private readonly double _trainSize;
    private readonly double _validationSize;

    /// <summary>
    /// Creates a new train/validation/test splitter.
    /// </summary>
    /// <param name="trainSize">Proportion for training (default 0.7 = 70%).</param>
    /// <param name="validationSize">Proportion for validation (default 0.15 = 15%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <remarks>
    /// Test size is automatically computed as 1 - trainSize - validationSize.
    /// With defaults: train=70%, validation=15%, test=15%.
    /// </remarks>
    public TrainValTestSplitter(
        double trainSize = 0.7,
        double validationSize = 0.15,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (trainSize <= 0 || trainSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trainSize),
                "Train size must be between 0 and 1 (exclusive).");
        }

        if (validationSize <= 0 || validationSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(validationSize),
                "Validation size must be between 0 and 1 (exclusive).");
        }

        if (trainSize + validationSize >= 1)
        {
            throw new ArgumentException(
                "Train size + validation size must be less than 1 to leave room for test set.");
        }

        _trainSize = trainSize;
        _validationSize = validationSize;
    }

    /// <inheritdoc/>
    public override bool SupportsValidation => true;

    /// <inheritdoc/>
    public override string Description
    {
        get
        {
            double testSize = 1 - _trainSize - _validationSize;
            return $"Train/Val/Test split ({_trainSize * 100:F0}%/{_validationSize * 100:F0}%/{testSize * 100:F0}%)";
        }
    }

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        var indices = GetShuffledIndices(nSamples);

        var (trainSize, validationSize, testSize) = ComputeSplitSizes(nSamples, _trainSize, _validationSize);

        var trainIndices = indices.Take(trainSize).ToArray();
        var validationIndices = indices.Skip(trainSize).Take(validationSize).ToArray();
        var testIndices = indices.Skip(trainSize + validationSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices, validationIndices);
    }

    /// <inheritdoc/>
    protected override (int[] TrainIndices, int[] TestIndices, int[]? ValidationIndices, int? FoldIndex, int? TotalFolds)
        SplitIndicesOnly(int nSamples, Vector<T>? y)
    {
        var indices = GetShuffledIndices(nSamples);

        var (trainSize, validationSize, _) = ComputeSplitSizes(nSamples, _trainSize, _validationSize);

        var trainIndices = indices.Take(trainSize).ToArray();
        var validationIndices = indices.Skip(trainSize).Take(validationSize).ToArray();
        var testIndices = indices.Skip(trainSize + validationSize).ToArray();

        return (trainIndices, testIndices, validationIndices, null, null);
    }
}
