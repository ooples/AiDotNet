using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Smoothly Clipped Absolute Deviation (SCAD) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SCAD is a nonconvex penalty that reduces bias for large coefficients while
/// maintaining sparsity. Unlike Lasso, SCAD doesn't overshrink large coefficients,
/// providing near-unbiased estimates for significant features.
/// </para>
/// <para><b>For Beginners:</b> SCAD is like an improved version of Lasso. Lasso
/// tends to shrink all coefficients, even important ones. SCAD lets truly important
/// features keep their full strength while still zeroing out unimportant ones.
/// This gives you better feature selection with less bias.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SCADSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly double _a;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SCADSelector(
        int nFeaturesToSelect = 10,
        double lambda = 1.0,
        double a = 3.7,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));
        if (a <= 2)
            throw new ArgumentException("SCAD parameter 'a' must be greater than 2.", nameof(a));

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
            "SCADSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var X = FeatureSelectionTensorOps.Standardize(Tensor<T>.FromMatrix(data), out _);
        var y = FeatureSelectionTensorOps.Center(Tensor<T>.FromVector(target));

        // Coordinate descent with SCAD penalty on the partial residual correlation x_j . r_j / n
        T count = NumOps.FromDouble(n);
        var coefficients = FeatureSelectionTensorOps.CoordinateDescent(X, y, _maxIterations, _tolerance,
            (partial, j) => FeatureSelectionTensorOps.ScadThreshold(engine.TensorDivideScalar(partial, count), _lambda, _a), out _);
        _coefficients = coefficients.ToVector();

        // Select top features by absolute coefficient, dropping (near-)zero ones unless none is left
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        var magnitude = engine.TensorAbs(coefficients);
        var nonZero = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            engine.TensorGreaterThan(magnitude, NumOps.FromDouble(1e-10))));
        _selectedIndices = PreprocessingTensorOps.StableOrder(magnitude, descending: true)
            .Take(nToSelect)
            .Where(j => nonZero.Contains(j))
            .OrderBy(x => x)
            .ToArray();

        if (_selectedIndices.Length == 0)
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(magnitude, nToSelect);
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
            throw new InvalidOperationException("SCADSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SCADSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SCADSelector has not been fitted.");

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
