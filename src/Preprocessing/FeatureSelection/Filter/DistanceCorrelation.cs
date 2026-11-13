using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Distance Correlation for detecting nonlinear relationships between features and target.
/// </summary>
/// <remarks>
/// <para>
/// Distance correlation measures both linear and nonlinear dependencies between variables.
/// Unlike Pearson correlation which only detects linear relationships, distance correlation
/// equals zero if and only if the variables are statistically independent.
/// </para>
/// <para><b>For Beginners:</b> Pearson correlation can miss relationships where one variable
/// depends on another in a curved or complex way. Distance correlation catches all types
/// of relationships. If it's zero, the variables are truly independent; if it's high,
/// there's some kind of relationship (linear or not).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DistanceCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCorrelation;

    private Vector<T>? _distanceCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DistanceCorrelations => _distanceCorrelations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DistanceCorrelation(
        int nFeaturesToSelect = 10,
        double minCorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minCorrelation = minCorrelation;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DistanceCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Distance correlation dCor = sqrt(dCov^2 / sqrt(dVar^2(x) dVar^2(y))) (Szekely, Rizzo and Bakirov 2007),
        // 0 when the denominator is not above 1e-10
        var dcov2 = FeatureSelectionTensorOps.SquaredDistanceCovariances(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target), out var xVariances, out T yVariance);
        var zeros = new Tensor<T>(new[] { p });
        // sqrt(a) sqrt(b) rather than sqrt(a b): the product of the two variances overflows Half's 65504 maximum
        var denominators = engine.TensorMultiplyScalar(engine.TensorSqrt(xVariances), NumOps.Sqrt(yVariance));
        var usable = engine.TensorGreaterThan(denominators, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var safe = engine.TensorWhere(usable, denominators, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        var distanceCorrelations = engine.TensorWhere(usable,
            engine.TensorSqrt(engine.TensorMax(engine.TensorDivide(dcov2, safe), NumOps.Zero)), zeros);
        _distanceCorrelations = distanceCorrelations.ToVector();

        // Select features above threshold or top by distance correlation
        var candidates = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.AtLeast(distanceCorrelations, NumOps.FromDouble(_minCorrelation))));

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = PreprocessingTensorOps.StableOrder(distanceCorrelations, descending: true)
                .Where(j => candidates.Contains(j))
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(distanceCorrelations, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("DistanceCorrelation has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DistanceCorrelation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DistanceCorrelation has not been fitted.");

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
