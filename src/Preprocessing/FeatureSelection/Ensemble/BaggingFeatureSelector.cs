using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Bagging-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses bootstrap aggregation (bagging) to create multiple subsamples and
/// aggregates feature importance across all subsamples for robust selection.
/// </para>
/// <para><b>For Beginners:</b> Bagging creates many random subsets of your data,
/// evaluates feature importance on each subset, then combines the results. This
/// helps ensure that the selected features are consistently important across
/// different parts of your data, not just important by chance.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BaggingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nEstimators;
    private readonly double _subsampleRatio;
    private readonly int? _randomState;

    private Vector<T>? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NEstimators => _nEstimators;
    public Vector<T>? AggregatedScores => _aggregatedScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BaggingFeatureSelector(
        int nFeaturesToSelect = 10,
        int nEstimators = 10,
        double subsampleRatio = 0.8,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nEstimators = nEstimators;
        _subsampleRatio = subsampleRatio;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BaggingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int subsampleSize = (int)(n * _subsampleRatio);

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var aggregatedScores = new Tensor<T>(new[] { p });

        for (int est = 0; est < _nEstimators; est++)
        {
            // Bootstrap sample
            var indices = Enumerable.Range(0, n)
                .OrderBy(_ => rand.Next())
                .Take(subsampleSize)
                .ToArray();

            // Compute correlation-based importance on subsample (0 for an empty subsample or a constant column)
            if (indices.Length > 0)
            {
                var rows = new Tensor<int>(indices, new[] { indices.Length });
                aggregatedScores = engine.TensorAdd(aggregatedScores, engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
                    engine.TensorIndexSelect(X, rows, 0), engine.TensorIndexSelect(y, rows, 0))));
            }
        }

        // Average the scores
        if (_nEstimators > 0)
            aggregatedScores = engine.TensorDivideScalar(aggregatedScores, NumOps.FromDouble(_nEstimators));
        _aggregatedScores = aggregatedScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(aggregatedScores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BaggingFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BaggingFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BaggingFeatureSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
