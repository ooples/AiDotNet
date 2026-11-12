using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Stability Selection based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their selection stability across multiple subsamples
/// of the data, ensuring robust feature selection.
/// </para>
/// <para><b>For Beginners:</b> Stability selection repeatedly subsamples the data
/// and performs feature selection on each subsample. Features that are consistently
/// selected across many subsamples are more reliable.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class StabilitySelectionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSubsamples;
    private readonly double _subsampleRatio;
    private readonly int? _randomState;

    private Vector<T>? _stabilityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NSubsamples => _nSubsamples;
    public double SubsampleRatio => _subsampleRatio;
    public Vector<T>? StabilityScores => _stabilityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public StabilitySelectionSelector(
        int nFeaturesToSelect = 10,
        int nSubsamples = 50,
        double subsampleRatio = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSubsamples = nSubsamples;
        _subsampleRatio = subsampleRatio;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StabilitySelectionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var selectionCounts = new int[p];
        int subsampleSize = (int)(n * _subsampleRatio);
        int selectPerSubsample = Math.Max(1, _nFeaturesToSelect / 2);

        for (int s = 0; s < _nSubsamples; s++)
        {
            // Create random subsample
            var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(subsampleSize).ToArray();

            // Compute |correlation| scores on subsample (0 below two rows or for a zero denominator)
            var rows = new Tensor<int>(indices, new[] { indices.Length });
            var scores = subsampleSize < 2
                ? PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero)
                : Engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
                    Engine.TensorIndexSelect(x, rows, 0), Engine.TensorIndexSelect(y, rows, 0)));

            // Select top features from this subsample
            foreach (int j in PreprocessingTensorOps.TopKIndicesSorted(scores, selectPerSubsample))
                selectionCounts[j]++;
        }

        // Stability scores = selection frequency
        var stability = Engine.TensorDivideScalar(
            new Tensor<int>(selectionCounts, new[] { p }).Cast<T>(), NumOps.FromDouble(_nSubsamples));
        _stabilityScores = stability.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(stability, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("StabilitySelectionSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("StabilitySelectionSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelectionSelector has not been fitted.");

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
