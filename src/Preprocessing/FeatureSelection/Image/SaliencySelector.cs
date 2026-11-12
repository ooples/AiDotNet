using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Image;

/// <summary>
/// Saliency-based feature selection for image features.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on gradient-based saliency, identifying features that
/// contribute most to model predictions. Originally designed for neural network
/// interpretability but adapted here for general feature selection.
/// </para>
/// <para><b>For Beginners:</b> Saliency measures how much a small change in each
/// feature would affect the prediction. Features with high saliency are "sensitive"
/// to changes, suggesting they're important for the model's decisions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SaliencySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _epsilon;

    private Vector<T>? _saliencyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SaliencyScores => _saliencyScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SaliencySelector(
        int nFeaturesToSelect = 100,
        double epsilon = 1e-5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _epsilon = epsilon;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SaliencySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // First, fit a simple linear model per feature: slope = Sxy / Sxx (0 when Sxx is not above 1e-10)
        var dx = engine.TensorSubtract(X, engine.ReduceMean(X, new[] { 0 }, true));
        var dy = engine.TensorSubtract(y, engine.ReduceMean(y, new[] { 0 }, true));
        var ssXY = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(dx), dy), new[] { p });
        var ssXX = engine.ReduceSum(engine.TensorSquare(dx), new[] { 0 }, false);
        var defined = engine.TensorGreaterThan(ssXX, NumOps.FromDouble(1e-10));
        var safeSsXX = engine.TensorWhere(defined, ssXX, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
        var coefficients = engine.TensorWhere(defined, engine.TensorDivide(ssXY, safeSsXX), new Tensor<T>(new[] { p }));

        // Saliency as the average absolute gradient x input, |coefficient| * mean |x|
        var meanAbs = engine.ReduceMean(engine.TensorAbs(X), new[] { 0 }, false);
        var saliency = engine.TensorMultiply(engine.TensorAbs(coefficients), meanAbs);
        _saliencyScores = saliency.ToVector();

        // Select top features by saliency
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(saliency, nToSelect);

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
            throw new InvalidOperationException("SaliencySelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SaliencySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SaliencySelector has not been fitted.");

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
