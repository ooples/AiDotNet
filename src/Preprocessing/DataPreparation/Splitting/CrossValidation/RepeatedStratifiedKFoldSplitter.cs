namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// Alias for <see cref="StratifiedRepeatedKFoldSplitter{T}"/> with a clearer name ordering.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the same as StratifiedRepeatedKFoldSplitter but with
/// a different name ordering that some users may find more intuitive.
/// </para>
/// <para>
/// Both names describe the same technique:
/// - RepeatedStratifiedKFold: "Repeated" (many times) + "Stratified" (class-preserving) + "K-Fold"
/// - StratifiedRepeatedKFold: "Stratified" (class-preserving) + "Repeated" (many times) + "K-Fold"
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RepeatedStratifiedKFoldSplitter<T> : StratifiedRepeatedKFoldSplitter<T>
{
    /// <summary>
    /// Creates a new Repeated Stratified K-Fold splitter.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="nRepeats">Number of repeats. Default is 10.</param>
    /// <param name="randomSeed">Base random seed.</param>
    public RepeatedStratifiedKFoldSplitter(int k = 5, int nRepeats = 10, int randomSeed = 42)
        : base(k, nRepeats, randomSeed)
    {
    }

    // Description inherited from base class (StratifiedRepeatedKFoldSplitter)
}
