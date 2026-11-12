using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Elastic Net regularization-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization. Like Lasso, it can
/// drive coefficients to zero for feature selection, but the L2 component helps when
/// features are correlated. The l1_ratio parameter controls the mix between L1 and L2.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net is a compromise between Lasso and Ridge
/// regression. When features are highly correlated, Lasso tends to pick one and ignore
/// the others. Elastic Net groups correlated features together, selecting them as a
/// group rather than arbitrarily picking one.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ElasticNetFeatureSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double L1Ratio => _l1Ratio;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ElasticNetFeatureSelection(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        double l1Ratio = 0.5,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        if (l1Ratio < 0 || l1Ratio > 1)
            throw new ArgumentException("L1 ratio must be between 0 and 1.", nameof(l1Ratio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _l1Ratio = l1Ratio;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ElasticNetFeatureSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Standardize features (population std; a near-zero std is replaced by 1) and center the target
        var X = FeatureSelectionTensorOps.Standardize(Tensor<T>.FromMatrix(data), out _);
        var y = FeatureSelectionTensorOps.Center(Tensor<T>.FromVector(target));

        // Coordinate descent for Elastic Net: beta_j = S(rho, l1) / (||x_j||^2 + l2), 0 when that denominator is
        // below 1e-10
        T l1Penalty = NumOps.FromDouble(_alpha * _l1Ratio * n);
        T l2Penalty = NumOps.FromDouble(_alpha * (1 - _l1Ratio) * n);
        T floor = NumOps.FromDouble(1e-10);
        var none = new Tensor<T>(new[] { 1 });
        var denominators = engine.TensorAddScalar(engine.ReduceSum(engine.TensorSquare(X), new[] { 0 }, false), l2Penalty);
        var coefficients = FeatureSelectionTensorOps.CoordinateDescent(X, y, _maxIterations, _tolerance, (rho, j) =>
        {
            var denom = engine.TensorSlice(denominators, new[] { j }, new[] { 1 });
            return engine.TensorWhere(engine.TensorLessThan(denom, floor), none,
                engine.TensorDivide(FeatureSelectionTensorOps.SoftThreshold(rho, l1Penalty), denom));
        }, out _);
        _coefficients = coefficients.ToVector();

        // Select features by absolute coefficient magnitude
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(engine.TensorAbs(coefficients), numToSelect);
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
            throw new InvalidOperationException("ElasticNetFeatureSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ElasticNetFeatureSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ElasticNetFeatureSelection has not been fitted.");

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
