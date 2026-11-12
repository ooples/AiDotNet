using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// L1 (Lasso) regularization-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Lasso regression uses L1 regularization which drives some coefficients exactly to zero,
/// effectively performing feature selection. Features with non-zero coefficients are selected.
/// The regularization strength controls how many features are zeroed out.
/// </para>
/// <para><b>For Beginners:</b> Lasso is like a strict editor that removes unnecessary words
/// from a sentence. The L1 penalty forces some feature weights to become exactly zero,
/// automatically removing unimportant features. Stronger regularization means more features
/// get eliminated, leaving only the most essential ones.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LassoFeatureSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
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
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LassoFeatureSelection(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LassoFeatureSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Coordinate descent for Lasso: soft-threshold rho at alpha * n and divide by ||x_j||^2
        // (a column with ||x_j||^2 below 1e-10 gets 0)
        var squaredNorms = engine.ReduceSum(engine.TensorSquare(X), new[] { 0 }, false);
        T threshold = NumOps.FromDouble(_alpha * n);
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var none = new Tensor<T>(new[] { 1 });
        var coefficients = FeatureSelectionTensorOps.CoordinateDescent(X, y, _maxIterations, _tolerance, (rho, j) =>
        {
            var sumXjSq = engine.TensorSlice(squaredNorms, new[] { j }, new[] { 1 });
            return engine.TensorWhere(engine.TensorLessThan(sumXjSq, floor), none,
                engine.TensorDivide(FeatureSelectionTensorOps.SoftThreshold(rho, threshold), sumXjSq));
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
            throw new InvalidOperationException("LassoFeatureSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LassoFeatureSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LassoFeatureSelection has not been fitted.");

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
