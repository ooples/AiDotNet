using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Elastic Net-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Elastic Net combines L1 (LASSO) and L2 (Ridge) regularization to select
/// features while handling correlated features better than pure LASSO.
/// Features with non-zero coefficients are selected.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net is like LASSO but smarter about groups
/// of related features. Instead of picking just one from a correlated group (like
/// LASSO does), it can keep several if they're all useful. This is great when you
/// have features that measure similar things.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ElasticNetFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
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
    public int MaxIterations => _maxIterations;
    public double Tolerance => _tolerance;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ElasticNetFS(
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
        if (alpha <= 0)
            throw new ArgumentException("Alpha must be positive.", nameof(alpha));
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
            "ElasticNetFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var xStandardized = FeatureScoringTensorOps.Standardize(Tensor<T>.FromMatrix(data), out var xStds);
        var yCentered = FeatureScoringTensorOps.Center(Tensor<T>.FromVector(target));

        // Coordinate descent for Elastic Net on the 1/(2n)-scaled objective (as in sklearn ElasticNet):
        // beta_j = S(x_j . r / n, alpha * l1Ratio) / (1 + alpha * (1 - l1Ratio)) for standardized x_j
        T count = NumOps.FromDouble(n);
        T l1Penalty = NumOps.FromDouble(_alpha * _l1Ratio);
        T l2Scale = NumOps.FromDouble(1 + _alpha * (1 - _l1Ratio));
        var coefficients = FeatureScoringTensorOps.CoordinateDescent(xStandardized, yCentered, _maxIterations, _tolerance,
            (rho, j) => engine.TensorDivideScalar(
                FeatureScoringTensorOps.SoftThreshold(engine.TensorDivideScalar(rho, count), l1Penalty), l2Scale), out _);

        // Rescale coefficients back to original scale
        coefficients = engine.TensorDivide(coefficients, xStds);
        _coefficients = coefficients.ToVector();

        // Select features with largest absolute coefficients
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
            throw new InvalidOperationException("ElasticNetFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ElasticNetFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ElasticNetFS has not been fitted.");

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
