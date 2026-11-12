using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Minimax Concave Penalty (MCP) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// MCP is a nonconvex penalty similar to SCAD that provides nearly unbiased
/// estimates for large coefficients while inducing sparsity. It has a concave
/// shape that reduces shrinkage bias.
/// </para>
/// <para><b>For Beginners:</b> Like SCAD, MCP is an improvement over Lasso.
/// It strongly penalizes small coefficients (pushing them to zero) but
/// gradually reduces the penalty for larger coefficients. This means truly
/// important features keep their full effect while noise features are eliminated.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MCPSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly double _gamma;
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

    public MCPSelector(
        int nFeaturesToSelect = 10,
        double lambda = 1.0,
        double gamma = 3.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));
        if (gamma <= 1)
            throw new ArgumentException("MCP parameter 'gamma' must be greater than 1.", nameof(gamma));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lambda = lambda;
        _gamma = gamma;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MCPSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Coordinate descent with MCP penalty on the partial residual correlation x_j . r_j / n
        T count = NumOps.FromDouble(n);
        var coefficients = FeatureSelectionTensorOps.CoordinateDescent(X, y, _maxIterations, _tolerance,
            (partial, j) => FeatureSelectionTensorOps.McpThreshold(engine.TensorDivideScalar(partial, count), _lambda, _gamma), out _);
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
            throw new InvalidOperationException("MCPSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MCPSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCPSelector has not been fitted.");

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
