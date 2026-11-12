using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Stability Selection for robust feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// Runs a base feature selector on many bootstrap samples and selects
/// features that are consistently chosen across subsamples. Provides
/// error rate control.
/// </para>
/// <para><b>For Beginners:</b> By running selection on many random subsets
/// of data, we find features that are "stably" selected regardless of
/// which samples we use. These are more likely to be truly important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class StabilitySelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _sampleFraction;
    private readonly double _threshold;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, int[]>? _baseSelector;

    private Vector<T>? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SelectionProbabilities => _selectionProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public StabilitySelection(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double sampleFraction = 0.5,
        double threshold = 0.6,
        Func<Matrix<T>, Vector<T>, int[]>? baseSelector = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBootstraps < 10)
            throw new ArgumentException("Number of bootstraps must be at least 10.", nameof(nBootstraps));
        if (sampleFraction <= 0 || sampleFraction > 1)
            throw new ArgumentException("Sample fraction must be between 0 and 1.", nameof(sampleFraction));
        if (threshold <= 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _sampleFraction = sampleFraction;
        _threshold = threshold;
        _baseSelector = baseSelector;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StabilitySelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int sampleSize = (int)(n * _sampleFraction);
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var selectionCounts = new Tensor<T>(new[] { p });

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Subsample without replacement (Meinshausen and Buehlmann 2010)
            var sampleIndices = Enumerable.Range(0, n)
                .OrderBy(_ => random.Next())
                .Take(sampleSize)
                .ToArray();
            var rows = new Tensor<int>(sampleIndices, new[] { sampleSize });
            var subMatrix = engine.TensorIndexSelect(x, rows, 0).ToMatrix();
            var subVector = engine.TensorIndexSelect(y, rows, 0).ToVector();

            // Run base selector
            int[] selected = _baseSelector is not null
                ? _baseSelector(subMatrix, subVector)
                : DefaultSelector(subMatrix, subVector, Math.Min(_nFeaturesToSelect, p));

            // Count selections (each feature at most once per subsample)
            var valid = selected.Where(idx => idx >= 0 && idx < p).ToArray();
            selectionCounts = engine.TensorAdd(selectionCounts, PopulationSearchOps.IndexMask<T>(valid, p));
        }

        // Compute selection probabilities
        var probabilities = engine.TensorDivideScalar(selectionCounts, NumOps.FromDouble(_nBootstraps));
        _selectionProbabilities = probabilities.ToVector();

        // Select features above threshold, falling back to the top ones by probability
        var stable = engine.TensorGreaterEqual(
            probabilities, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.FromDouble(_threshold)));
        _selectedIndices = FeatureSelectionTensorOps.TopKWhere(probabilities, stable, _nFeaturesToSelect);
        if (_selectedIndices.Length == 0)
            _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(probabilities, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    private int[] DefaultSelector(Matrix<T> data, Vector<T> target, int k) =>
        FeatureSelectionTensorOps.TopKIndicesSorted(PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero), k);

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
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("StabilitySelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
