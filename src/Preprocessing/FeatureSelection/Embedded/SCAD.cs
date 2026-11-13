using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Smoothly Clipped Absolute Deviation (SCAD) penalty for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SCAD is a non-convex penalty that addresses the bias problem of LASSO.
/// It applies the same penalty as LASSO for small coefficients but reduces
/// the penalty for large coefficients, leading to nearly unbiased estimates.
/// </para>
/// <para><b>For Beginners:</b> LASSO tends to shrink large coefficients too much
/// (bias). SCAD fixes this by penalizing large coefficients less aggressively
/// while still shrinking small ones to zero. This gives you better estimates
/// of important feature effects.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SCAD<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly double _a; // SCAD parameter (typically 3.7)
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Lambda => _lambda;
    public double A => _a;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SCAD(
        int nFeaturesToSelect = 10,
        double lambda = 0.1,
        double a = 3.7,
        int maxIterations = 100,
        double tolerance = 1e-6,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));
        if (a <= 2)
            throw new ArgumentException("Parameter 'a' must be greater than 2.", nameof(a));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lambda = lambda;
        _a = a;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SCAD requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Standardize data (population std; a near-zero std is replaced by 1) and center the target
        var X = FeatureScoringTensorOps.Standardize(Tensor<T>.FromMatrix(data), out _);
        var y = FeatureScoringTensorOps.Center(Tensor<T>.FromVector(target));

        // Coordinate descent with SCAD penalty on the partial residual correlation rho = x_j . r_j / n
        T count = NumOps.FromDouble(n);
        var coefficients = FeatureScoringTensorOps.CoordinateDescent(X, y, _maxIterations, _tolerance,
            (rho, j) => FeatureScoringTensorOps.ScadThreshold(engine.TensorDivideScalar(rho, count), _lambda, _a), out _);
        _coefficients = coefficients.ToVector();

        // Select features with non-zero coefficients (largest first); fall back to the top correlations
        var magnitude = engine.TensorAbs(coefficients);
        var nonZero = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            engine.TensorGreaterThan(magnitude, PreprocessingTensorOps.Tolerance<T>(1e-10))));
        if (nonZero.Count == 0)
        {
            var correlations = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
                Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)));
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(correlations, Math.Min(_nFeaturesToSelect, p));
        }
        else
        {
            _selectedIndices = PreprocessingTensorOps.StableOrder(magnitude, descending: true)
                .Where(j => nonZero.Contains(j))
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
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
            throw new InvalidOperationException("SCAD has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SCAD does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SCAD has not been fitted.");

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
