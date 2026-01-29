namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Basic;

/// <summary>
/// Alias for TrainTestSplitter for users who prefer the "Random" naming convention.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is exactly the same as TrainTestSplitter.
/// Different libraries use different names for the same operation:
/// - scikit-learn: train_test_split
/// - Some libraries: random_split
/// - AiDotNet: Both TrainTestSplitter and RandomSplitter
/// </para>
/// <para>
/// Use whichever name feels more natural to you!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RandomSplitter<T> : TrainTestSplitter<T>
{
    /// <summary>
    /// Creates a new random splitter (alias for train/test splitter).
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public RandomSplitter(double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
        : base(testSize, shuffle, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override string Description => $"Random split ({(1 - 0.2) * 100:F0}% train, {0.2 * 100:F0}% test)";
}
