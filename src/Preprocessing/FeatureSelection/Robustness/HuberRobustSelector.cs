using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Robustness;

/// <summary>
/// Huber Robust Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their coefficients in a Huber regression, which
/// downweights the influence of outliers using the Huber loss function.
/// </para>
/// <para><b>For Beginners:</b> The Huber loss combines squared error for small
/// residuals with absolute error for large ones. This makes the regression less
/// sensitive to outliers while still being efficient for normal data.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HuberRobustSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _delta;
    private readonly int _maxIterations;
    private readonly double _learningRate;

    private Vector<T>? _huberCoefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Delta => _delta;
    public Vector<T>? HuberCoefficients => _huberCoefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public HuberRobustSelector(
        int nFeaturesToSelect = 10,
        double delta = 1.35,
        int maxIterations = 100,
        double learningRate = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _delta = delta;
        _maxIterations = maxIterations;
        _learningRate = learningRate;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HuberRobustSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Standardize features and target (population std; near-constant ones keep scale 1)
        var x = SelectorTensorOps.StandardizeColumns(Tensor<T>.FromMatrix(data));
        var y = SelectorTensorOps.StandardizeColumns(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }));
        var xt = engine.TensorTranspose(x);

        // Fit Huber regression via gradient descent
        var weights = PreprocessingTensorOps.Filled(new[] { p, 1 }, NumOps.Zero);
        var bias = PreprocessingTensorOps.Filled(new[] { 1, 1 }, NumOps.Zero);
        T delta = NumOps.FromDouble(_delta);
        T negativeDelta = NumOps.Negate(delta);
        T step = NumOps.FromDouble(_learningRate / n);
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var residual = engine.TensorSubtract(engine.TensorAdd(engine.TensorMatMul(x, weights), bias), y);

            // Huber loss gradient: the residual, clipped to [-delta, delta]
            var factor = engine.TensorClamp(residual, negativeDelta, delta);
            weights = engine.TensorSubtract(weights, engine.TensorMultiplyScalar(engine.TensorMatMul(xt, factor), step));
            bias = engine.TensorSubtract(bias, engine.TensorMultiplyScalar(engine.ReduceSum(factor, new[] { 0 }, true), step));
        }

        var coefficients = engine.TensorAbs(engine.Reshape(weights, new[] { p }));
        _huberCoefficients = coefficients.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(coefficients, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("HuberRobustSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HuberRobustSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HuberRobustSelector has not been fitted.");

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
