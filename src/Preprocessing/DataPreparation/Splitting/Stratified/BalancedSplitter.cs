using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified;

/// <summary>
/// Balanced splitter that ensures equal representation of each class in train and test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When you have imbalanced classes (e.g., 90% normal, 10% fraud),
/// this splitter ensures both train and test sets have equal numbers of each class
/// by undersampling the majority classes.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Find the smallest class size
/// 2. Sample equally from each class to match the smallest
/// 3. Split the balanced data into train/test
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you want to evaluate model performance without class imbalance effects
/// - When minority class performance is critical
/// - For initial model development before handling imbalance in other ways
/// </para>
/// <para>
/// <b>Caution:</b> This discards data from majority classes. For production,
/// consider SMOTE or other oversampling techniques instead.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class BalancedSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;

    /// <summary>
    /// Creates a new balanced splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public BalancedSplitter(double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Balanced split ({_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y), "Balanced split requires target labels (y).");
        }

        // Group indices by class label
        var classSamples = GroupByLabel(y);

        if (classSamples.Count < 2)
        {
            throw new ArgumentException("Balanced split requires at least 2 classes.");
        }

        // Find minimum class size
        int minClassSize = classSamples.Values.Min(list => list.Count);

        if (minClassSize < 2)
        {
            throw new ArgumentException(
                "All classes must have at least 2 samples for balanced splitting.");
        }

        // Calculate samples per class for train and test
        int testPerClass = Math.Max(1, (int)(minClassSize * _testSize));
        int trainPerClass = minClassSize - testPerClass;

        if (trainPerClass < 1)
        {
            trainPerClass = 1;
            testPerClass = minClassSize - 1;
        }

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        foreach (var kvp in classSamples)
        {
            var indices = kvp.Value.ToArray();

            if (_shuffle)
            {
                ShuffleIndices(indices);
            }

            // Take equal samples from each class
            for (int i = 0; i < trainPerClass; i++)
            {
                trainIndices.Add(indices[i]);
            }

            for (int i = trainPerClass; i < trainPerClass + testPerClass; i++)
            {
                testIndices.Add(indices[i]);
            }
        }

        // Shuffle final indices if requested
        if (_shuffle)
        {
            var trainArray = trainIndices.ToArray();
            var testArray = testIndices.ToArray();
            ShuffleIndices(trainArray);
            ShuffleIndices(testArray);
            return BuildResult(X, y, trainArray, testArray);
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
