using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// LASSO (Least Absolute Shrinkage and Selection Operator) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// LASSO uses L1 regularization during linear regression training, which naturally
/// drives some coefficients to exactly zero. Features with non-zero coefficients
/// are selected for the final model.
/// </para>
/// <para><b>For Beginners:</b> LASSO is a special kind of linear regression that
/// automatically picks only the most important features. It does this by adding
/// a penalty that forces unimportant feature weights to become exactly zero,
/// effectively removing them from the model.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LassoFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public int MaxIterations => _maxIterations;
    public double Tolerance => _tolerance;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LassoFS(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha <= 0)
            throw new ArgumentException("Alpha must be positive.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LassoFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var xStandardized = FeatureSelectionTensorOps.Standardize(Tensor<T>.FromMatrix(data), out var xStds);
        var yCentered = FeatureSelectionTensorOps.Center(Tensor<T>.FromVector(target));

        // Coordinate descent for LASSO: rho = x_j . r / n, soft-thresholded at alpha
        T count = NumOps.FromDouble(n);
        T alpha = NumOps.FromDouble(_alpha);
        var coefficients = FeatureSelectionTensorOps.CoordinateDescent(xStandardized, yCentered, _maxIterations, _tolerance,
            (rho, j) => FeatureSelectionTensorOps.SoftThreshold(engine.TensorDivideScalar(rho, count), alpha), out _);

        // Rescale coefficients
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
            throw new InvalidOperationException("LassoFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LassoFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LassoFS has not been fitted.");

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
