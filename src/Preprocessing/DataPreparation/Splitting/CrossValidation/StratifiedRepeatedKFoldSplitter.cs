using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// Stratified Repeated K-Fold cross-validation combining stratification with multiple repeats.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This combines the benefits of:
/// - <b>Stratification:</b> Preserves class distribution in each fold
/// - <b>Repetition:</b> Runs multiple times for more stable estimates
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Classification problems with need for very reliable estimates
/// - Imbalanced datasets where stratification is important
/// - Model comparison where small differences matter
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StratifiedRepeatedKFoldSplitter<T> : DataSplitterBase<T>
{
    private readonly int _k;
    private readonly int _nRepeats;

    /// <summary>
    /// Creates a new Stratified Repeated K-Fold splitter.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="nRepeats">Number of repeats. Default is 10.</param>
    /// <param name="randomSeed">Base random seed.</param>
    public StratifiedRepeatedKFoldSplitter(int k = 5, int nRepeats = 10, int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "Number of folds must be at least 2.");
        }

        if (nRepeats < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nRepeats), "Number of repeats must be at least 1.");
        }

        _k = k;
        _nRepeats = nRepeats;
    }

    /// <inheritdoc/>
    public override int NumSplits => _k * _nRepeats;

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Stratified Repeated {_k}-Fold CV ({_nRepeats} repeats)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        for (int repeat = 0; repeat < _nRepeats; repeat++)
        {
            var stratifiedSplitter = new StratifiedKFoldSplitter<T>(_k, shuffle: true, randomSeed: _randomSeed + repeat);

            foreach (var split in stratifiedSplitter.GetSplits(X, y))
            {
                yield return new DataSplitResult<T>
                {
                    XTrain = split.XTrain,
                    XTest = split.XTest,
                    yTrain = split.yTrain,
                    yTest = split.yTest,
                    TrainIndices = split.TrainIndices,
                    TestIndices = split.TestIndices,
                    FoldIndex = split.FoldIndex,
                    TotalFolds = split.TotalFolds,
                    RepeatIndex = repeat,
                    TotalRepeats = _nRepeats
                };
            }
        }
    }
}
