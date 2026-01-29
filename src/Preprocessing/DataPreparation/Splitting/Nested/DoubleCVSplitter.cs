using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Nested;

/// <summary>
/// Double Cross-Validation splitter (alias for Nested CV with equal inner/outer folds).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Double CV is a common configuration of Nested CV where both
/// the inner and outer loops use the same number of folds (typically 5 or 10).
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you need unbiased performance estimation during hyperparameter search
/// - For comparing different model families fairly
/// - When computational resources allow (k² model fits)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DoubleCVSplitter<T> : DataSplitterBase<T>
{
    private readonly int _folds;

    /// <summary>
    /// Creates a new Double CV splitter.
    /// </summary>
    /// <param name="folds">Number of folds for both inner and outer loops. Default is 5.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DoubleCVSplitter(int folds = 5, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (folds < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(folds), "Number of folds must be at least 2.");
        }

        _folds = folds;
    }

    /// <inheritdoc/>
    public override int NumSplits => _folds * _folds;

    /// <inheritdoc/>
    public override bool SupportsValidation => true;

    /// <inheritdoc/>
    public override string Description => $"Double {_folds}-Fold CV";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        var indices = GetShuffledIndices(nSamples);

        int baseFoldSize = nSamples / _folds;
        int remainder = nSamples % _folds;

        int outerCurrentIndex = 0;
        int splitIndex = 0;

        for (int outerFold = 0; outerFold < _folds; outerFold++)
        {
            int outerFoldSize = baseFoldSize + (outerFold < remainder ? 1 : 0);

            var outerTestIndices = indices.Skip(outerCurrentIndex).Take(outerFoldSize).ToArray();

            var trainValIndices = new List<int>();
            for (int i = 0; i < outerCurrentIndex; i++)
            {
                trainValIndices.Add(indices[i]);
            }
            for (int i = outerCurrentIndex + outerFoldSize; i < nSamples; i++)
            {
                trainValIndices.Add(indices[i]);
            }

            var trainValArray = trainValIndices.ToArray();
            int innerNSamples = trainValArray.Length;

            int innerBaseFoldSize = innerNSamples / _folds;
            int innerRemainder = innerNSamples % _folds;

            int innerCurrentIndex = 0;
            for (int innerFold = 0; innerFold < _folds; innerFold++)
            {
                int innerFoldSize = innerBaseFoldSize + (innerFold < innerRemainder ? 1 : 0);

                var validationIndices = trainValArray.Skip(innerCurrentIndex).Take(innerFoldSize).ToArray();

                var trainIndices = new List<int>();
                for (int i = 0; i < innerCurrentIndex; i++)
                {
                    trainIndices.Add(trainValArray[i]);
                }
                for (int i = innerCurrentIndex + innerFoldSize; i < innerNSamples; i++)
                {
                    trainIndices.Add(trainValArray[i]);
                }

                yield return BuildResult(X, y, trainIndices.ToArray(), outerTestIndices, validationIndices,
                    foldIndex: splitIndex, totalFolds: NumSplits);

                innerCurrentIndex += innerFoldSize;
                splitIndex++;
            }

            outerCurrentIndex += outerFoldSize;
        }
    }
}
