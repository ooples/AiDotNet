using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// Repeated K-Fold cross-validation that runs K-Fold multiple times with different random seeds.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Repeated K-Fold runs K-Fold cross-validation multiple times,
/// each time with a different random shuffle. This gives even more stable performance estimates.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// <code>
/// Repeat 1: Run 5-Fold CV (5 evaluations)
/// Repeat 2: Shuffle differently, run 5-Fold CV (5 more evaluations)
/// Repeat 3: Shuffle differently, run 5-Fold CV (5 more evaluations)
/// Total: 15 evaluations
/// </code>
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you need very reliable performance estimates
/// - For statistical significance testing
/// - When comparing models and small differences matter
/// </para>
/// <para>
/// <b>Cost:</b> Runs k × n_repeats evaluations, so it's n_repeats times slower than regular K-Fold.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RepeatedKFoldSplitter<T> : DataSplitterBase<T>
{
    private readonly int _k;
    private readonly int _nRepeats;

    /// <summary>
    /// Creates a new Repeated K-Fold cross-validation splitter.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="nRepeats">Number of times to repeat K-Fold. Default is 10.</param>
    /// <param name="randomSeed">Base random seed. Each repeat uses a different derived seed.</param>
    public RepeatedKFoldSplitter(int k = 5, int nRepeats = 10, int randomSeed = 42)
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
    public override string Description => $"Repeated {_k}-Fold CV ({_nRepeats} repeats, {NumSplits} total evaluations)";

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
            // Create a new K-Fold splitter with a different seed for each repeat
            var kfoldSplitter = new KFoldSplitter<T>(_k, shuffle: true, randomSeed: _randomSeed + repeat);

            foreach (var split in kfoldSplitter.GetSplits(X, y))
            {
                // Re-wrap with repeat information
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
