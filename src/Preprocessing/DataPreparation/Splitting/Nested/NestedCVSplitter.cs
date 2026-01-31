using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Nested;

/// <summary>
/// Nested (Double) Cross-Validation splitter for unbiased model selection and evaluation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Regular cross-validation can give biased results if you use it
/// for both model selection (hyperparameter tuning) AND performance estimation.
/// Nested CV solves this with two levels of cross-validation.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// - Outer loop: Splits data into train+val and test for final evaluation
/// - Inner loop: Splits train+val for model selection/hyperparameter tuning
/// </para>
/// <para>
/// <b>Visual Example (3-outer, 2-inner):</b>
/// <code>
/// Outer Fold 1: [Train+Val    ][Train+Val    ][Test]
///               └── Inner: [Train][Val], [Val][Train]
/// Outer Fold 2: [Train+Val    ][Test         ][Train+Val]
///               └── Inner: [Train][Val], [Val][Train]
/// Outer Fold 3: [Test         ][Train+Val    ][Train+Val]
///               └── Inner: [Train][Val], [Val][Train]
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class NestedCVSplitter<T> : DataSplitterBase<T>
{
    private readonly int _outerFolds;
    private readonly int _innerFolds;

    /// <summary>
    /// Creates a new Nested CV splitter.
    /// </summary>
    /// <param name="outerFolds">Number of outer folds for evaluation. Default is 5.</param>
    /// <param name="innerFolds">Number of inner folds for model selection. Default is 3.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public NestedCVSplitter(int outerFolds = 5, int innerFolds = 3, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (outerFolds < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(outerFolds), "Outer folds must be at least 2.");
        }

        if (innerFolds < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(innerFolds), "Inner folds must be at least 2.");
        }

        _outerFolds = outerFolds;
        _innerFolds = innerFolds;
    }

    /// <inheritdoc/>
    public override int NumSplits => _outerFolds * _innerFolds;

    /// <inheritdoc/>
    public override bool SupportsValidation => true;

    /// <inheritdoc/>
    public override string Description => $"Nested CV ({_outerFolds} outer x {_innerFolds} inner folds)";

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

        // Outer fold sizes
        int outerBaseFoldSize = nSamples / _outerFolds;
        int outerRemainder = nSamples % _outerFolds;

        int outerCurrentIndex = 0;
        int splitIndex = 0;

        for (int outerFold = 0; outerFold < _outerFolds; outerFold++)
        {
            int outerFoldSize = outerBaseFoldSize + (outerFold < outerRemainder ? 1 : 0);

            // Outer test indices
            var outerTestIndices = indices.Skip(outerCurrentIndex).Take(outerFoldSize).ToArray();

            // Outer train+val indices (everything except outer test)
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

            // Inner fold sizes
            int innerBaseFoldSize = innerNSamples / _innerFolds;
            int innerRemainder = innerNSamples % _innerFolds;

            int innerCurrentIndex = 0;
            for (int innerFold = 0; innerFold < _innerFolds; innerFold++)
            {
                int innerFoldSize = innerBaseFoldSize + (innerFold < innerRemainder ? 1 : 0);

                // Inner validation indices (for this inner fold)
                var validationIndices = trainValArray.Skip(innerCurrentIndex).Take(innerFoldSize).ToArray();

                // Inner train indices
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
