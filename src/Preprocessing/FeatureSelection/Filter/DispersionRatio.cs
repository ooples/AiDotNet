using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Dispersion Ratio for comparing within-class to between-class variation.
/// </summary>
/// <remarks>
/// <para>
/// The dispersion ratio measures the ratio of between-class variance to within-class
/// variance. Features with high dispersion ratios have well-separated class means
/// relative to the variation within each class.
/// </para>
/// <para><b>For Beginners:</b> A good feature for classification should have class
/// groups that are far apart (high between-class variance) and tight clusters
/// (low within-class variance). The dispersion ratio captures this by dividing
/// the separation by the spread. High values mean easy-to-separate classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DispersionRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minRatio;

    private Vector<T>? _dispersionRatios;
    private Vector<T>? _betweenClassVariances;
    private Vector<T>? _withinClassVariances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DispersionRatios => _dispersionRatios;
    public Vector<T>? BetweenClassVariances => _betweenClassVariances;
    public Vector<T>? WithinClassVariances => _withinClassVariances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DispersionRatio(
        int nFeaturesToSelect = 10,
        double minRatio = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minRatio = minRatio;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DispersionRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Group samples by class (rounded target values)
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        if (k < 2)
            throw new ArgumentException("At least 2 classes are required.");

        // Between-class variance sum_c n_c (mean_c - mean)^2 / n and within-class variance sum (x - mean_class)^2 / n
        T count = NumOps.FromDouble(n);
        var classMeans = FeatureSelectionTensorOps.GroupMeans(PreprocessingTensorOps.ClassOneHot<T>(classIndex, k), X, out var classCounts);
        var overallMean = engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { k, p });
        var betweenVar = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorMultiply(
            engine.TensorSquare(engine.TensorSubtract(classMeans, overallMean)),
            engine.TensorBroadcastTo(engine.Reshape(classCounts, new[] { k, 1 }), new[] { k, p })), new[] { 0 }, false), count);
        var withinVar = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(X,
            engine.TensorIndexSelect(classMeans, new Tensor<int>(classIndex, new[] { n }), 0))), new[] { 0 }, false), count);
        _betweenClassVariances = betweenVar.ToVector();
        _withinClassVariances = withinVar.ToVector();

        // Dispersion ratio (0 when the within-class variance is not above 1e-10)
        var usable = engine.TensorGreaterThan(withinVar, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var dispersionRatios = engine.TensorWhere(usable, engine.TensorDivide(betweenVar,
            engine.TensorWhere(usable, withinVar, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One))), new Tensor<T>(new[] { p }));
        _dispersionRatios = dispersionRatios.ToVector();

        // Select features above threshold or top by ratio
        var candidates = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.AtLeast(dispersionRatios, NumOps.FromDouble(_minRatio))));

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = PreprocessingTensorOps.StableOrder(dispersionRatios, descending: true)
                .Where(j => candidates.Contains(j))
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(dispersionRatios, Math.Min(_nFeaturesToSelect, p));
        }

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
            throw new InvalidOperationException("DispersionRatio has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DispersionRatio does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DispersionRatio has not been fitted.");

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
