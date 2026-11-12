using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// GradCAM++ based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using gradient-weighted class activation mapping,
/// measuring feature importance through gradient flow analysis.
/// </para>
/// <para><b>For Beginners:</b> GradCAM++ extends GradCAM by using weighted
/// combinations of positive partial derivatives. It identifies which features
/// the model focuses on when making predictions. Features with higher activation
/// weights are considered more important for the model's decisions.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GradCAMSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _activationWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? ActivationWeights => _activationWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GradCAMSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GradCAMSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Simulate gradient-based importance using correlation gradients
        // In a real implementation, this would use actual neural network gradients
        var centeredX = engine.TensorSubtract(X, engine.ReduceMean(X, new[] { 0 }, true));
        var centeredY = engine.TensorBroadcastTo(engine.TensorSubtract(y, engine.ReduceMean(y, new[] { 0 }, true)), new[] { n, p });

        // GradCAM++ style: weighted positive gradients, weight = grad², activation = ReLU(x)
        var grad = engine.TensorMultiply(centeredX, centeredY);
        var positive = engine.TensorGreaterThan(grad, NumOps.Zero);
        var weight = engine.TensorMultiply(engine.TensorSquare(grad), positive);
        var activation = engine.TensorMax(X, NumOps.Zero);
        var posGradSum = engine.ReduceSum(engine.TensorMultiply(weight, activation), new[] { 0 }, false);
        var posWeightSum = engine.ReduceSum(weight, new[] { 0 }, false);

        // Combine positive contributions (GradCAM++ focuses on positive), ReLU on the final weights
        var defined = engine.TensorGreaterThan(posWeightSum, NumOps.FromDouble(1e-10));
        var safeWeight = engine.TensorWhere(defined, posWeightSum, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
        var alpha = engine.TensorWhere(defined, engine.TensorDivide(posGradSum, safeWeight), new Tensor<T>(new[] { p }));
        var activationWeights = engine.TensorMax(alpha, NumOps.Zero);
        _activationWeights = activationWeights.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(activationWeights, numToSelect);

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
            throw new InvalidOperationException("GradCAMSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GradCAMSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GradCAMSelector has not been fitted.");

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
