using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Classification;

/// <summary>
/// Naive Bayes based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their discriminative power using Naive Bayes
/// class-conditional probability estimates.
/// </para>
/// <para><b>For Beginners:</b> Naive Bayes assumes features are independent given
/// the class. This selector measures how much each feature helps distinguish
/// between classes by looking at how different the feature distributions are
/// for each class.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NaiveBayesSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _discriminativePower;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DiscriminativePower => _discriminativePower;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public NaiveBayesSelector(
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
            "NaiveBayesSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        // Discriminative power: mean symmetric Gaussian KL divergence over every pair of classes.
        var power = new Tensor<T>(new[] { p });
        if (k >= 2)
        {
            // Class-conditional means and (population) variances, floored at 1e-10.
            var Y = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
            var Yt = engine.TensorTranspose(Y);
            // Class sizes counted as int: a 0/1 sum in Half stops at 2048
            var counts = engine.Reshape(engine.ReduceSum(Y.Cast<int>(), new[] { 0 }, false), new[] { k, 1 }).Cast<T>();
            var means = engine.TensorDivide(engine.TensorMatMul(Yt, X), counts);
            var centered = engine.TensorSubtract(X, engine.TensorMatMul(Y, means));
            var variances = engine.TensorMax(
                engine.TensorDivide(engine.TensorMatMul(Yt, engine.TensorSquare(centered)), counts),
                NumOps.FromDouble(1e-10));

            // Every ordered class pair (c1, c2) as [k, k, p]; the symmetric KL is symmetric in the pair and
            // exactly 0 on the diagonal, so the sum over c1 < c2 is half the sum over all pairs.
            var pairShape = new[] { k, k, p };
            var mean1 = engine.TensorBroadcastTo(engine.Reshape(means, new[] { k, 1, p }), pairShape);
            var mean2 = engine.TensorBroadcastTo(engine.Reshape(means, new[] { 1, k, p }), pairShape);
            var var1 = engine.TensorBroadcastTo(engine.Reshape(variances, new[] { k, 1, p }), pairShape);
            var var2 = engine.TensorBroadcastTo(engine.Reshape(variances, new[] { 1, k, p }), pairShape);
            var symmetricKl = ComputeGaussianKL(mean1, var1, mean2, var2);

            // Divided by k and then k - 1: the ordered pair count k (k - 1) overflows Half past k = 256
            power = engine.TensorDivideScalar(engine.TensorDivideScalar(
                engine.ReduceSum(symmetricKl, new[] { 0, 1 }, false), NumOps.FromDouble(k)), NumOps.FromDouble(k - 1));
        }

        _discriminativePower = power.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(power, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeGaussianKL(Tensor<T> mean1, Tensor<T> var1, Tensor<T> mean2, Tensor<T> var2)
    {
        // Symmetric KL divergence between two Gaussians, element-wise
        var engine = Engine;
        T half = NumOps.FromDouble(0.5);
        var meanDiffSq = engine.TensorSquare(engine.TensorSubtract(mean1, mean2));
        var kl12 = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorAdd(engine.TensorAdd(
            engine.TensorLog(engine.TensorDivide(var2, var1)), engine.TensorDivide(var1, var2)),
            engine.TensorDivide(meanDiffSq, var2)), NumOps.One), half);
        var kl21 = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorAdd(engine.TensorAdd(
            engine.TensorLog(engine.TensorDivide(var1, var2)), engine.TensorDivide(var2, var1)),
            engine.TensorDivide(meanDiffSq, var1)), NumOps.One), half);
        return engine.TensorMultiplyScalar(engine.TensorAdd(kl12, kl21), half);
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
            throw new InvalidOperationException("NaiveBayesSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("NaiveBayesSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NaiveBayesSelector has not been fitted.");

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
