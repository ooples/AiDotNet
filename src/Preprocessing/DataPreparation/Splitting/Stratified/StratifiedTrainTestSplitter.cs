using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified;

/// <summary>
/// Stratified train/test split that preserves class distribution.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A stratified split ensures that both your training and test sets
/// have the same proportion of each class as the original data.
/// </para>
/// <para>
/// <b>Why This Matters:</b>
/// Imagine your data has 90% cats and 10% dogs. With a random split, you might get
/// unlucky and have 95% cats in training but only 70% cats in test. This would
/// make your model evaluation unreliable. Stratification prevents this.
/// </para>
/// <para>
/// <b>Industry Standard:</b> For classification tasks, ALWAYS use stratified splitting
/// unless you have a specific reason not to.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StratifiedTrainTestSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;

    /// <summary>
    /// Creates a new stratified train/test splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle within classes. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public StratifiedTrainTestSplitter(double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
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
    public override string Description => $"Stratified Train/Test split ({(1 - _testSize) * 100:F0}%/{_testSize * 100:F0}%)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y), "Stratified split requires target labels (y).");
        }

        var labelGroups = GroupByLabel(y);
        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        foreach (var group in labelGroups)
        {
            var classIndices = group.Value.ToArray();

            if (_shuffle)
            {
                ShuffleIndices(classIndices);
            }

            int classTestSize = Math.Max(1, (int)(classIndices.Length * _testSize));
            int classTrainSize = classIndices.Length - classTestSize;

            // Ensure at least 1 sample in each set for each class
            if (classTrainSize < 1)
            {
                classTrainSize = 1;
                classTestSize = classIndices.Length - 1;
            }

            for (int i = 0; i < classTrainSize; i++)
            {
                trainIndices.Add(classIndices[i]);
            }

            for (int i = classTrainSize; i < classIndices.Length; i++)
            {
                testIndices.Add(classIndices[i]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
