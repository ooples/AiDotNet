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

        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        // Half and BFloat16 are scored in float and the scores cast back once: a squared mean difference of 300
        // (90000) and the sum of the pair divergences pass Half's 65504 though their average need not
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        if (PreprocessingTensorOps.IsNarrowFloat<T>())
        {
            var widePower = PairwiseKlPower(X.Cast<float>(), classIndex, k);
            _discriminativePower = widePower.Cast<T>().ToVector();
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(widePower, numToSelect);
        }
        else
        {
            var power = PairwiseKlPower(X, classIndex, k);
            _discriminativePower = power.ToVector();
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(power, numToSelect);
        }

        IsFitted = true;
    }

    /// <summary>
    /// Discriminative power of every column of <paramref name="X"/> ([n, p]): the mean symmetric Gaussian KL
    /// divergence over every unordered pair of the <paramref name="k"/> classes (0 with fewer than two classes).
    /// </summary>
    private static Tensor<TC> PairwiseKlPower<TC>(Tensor<TC> X, int[] classIndex, int k)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        int p = X.Shape[1];
        var power = new Tensor<TC>(new[] { p });
        if (k < 2)
            return power;

        // Class-conditional means and (population) variances, floored at 1e-10.
        var Y = PreprocessingTensorOps.ClassOneHot<TC>(classIndex, k);
        // Per-class means of the values and of the squared deviations through GroupMeans, which counts and sums
        // in double: a 0/1 count in Half stops at 2048
        var means = FeatureScoringTensorOps.GroupMeans(Y, X, out _);
        var centered = engine.TensorSubtract(X, engine.TensorMatMul(Y, means));
        var variances = engine.TensorMax(
            FeatureScoringTensorOps.GroupMeans(Y, engine.TensorSquare(centered), out _),
            PreprocessingTensorOps.Tolerance<TC>(1e-10));

        // Class c1 against every later class c2 > c1 at a time ([k - c1 - 1, p]), accumulated into one [p] sum: all
        // ordered pairs at once is a [k, k, p] tensor per intermediate (1.6 GB of doubles for 200 classes of 5000
        // features)
        for (int c1 = 0; c1 < k - 1; c1++)
        {
            int rest = k - c1 - 1;
            var restShape = new[] { rest, p };
            var mean1 = engine.TensorBroadcastTo(engine.TensorSlice(means, new[] { c1, 0 }, new[] { 1, p }), restShape);
            var var1 = engine.TensorBroadcastTo(engine.TensorSlice(variances, new[] { c1, 0 }, new[] { 1, p }), restShape);
            var mean2 = engine.TensorSlice(means, new[] { c1 + 1, 0 }, restShape);
            var var2 = engine.TensorSlice(variances, new[] { c1 + 1, 0 }, restShape);
            power = engine.TensorAdd(power,
                engine.ReduceSum(ComputeGaussianKL(mean1, var1, mean2, var2), new[] { 0 }, false));
        }

        // The mean over the k (k - 1) / 2 unordered pairs, divided by k and then (k - 1) / 2 so the pair count never
        // exists in TC
        var ops = MathHelper.GetNumericOperations<TC>();
        return engine.TensorDivideScalar(
            engine.TensorDivideScalar(power, ops.FromDouble(k)), ops.FromDouble((k - 1) / 2.0));
    }

    private static Tensor<TC> ComputeGaussianKL<TC>(Tensor<TC> mean1, Tensor<TC> var1, Tensor<TC> mean2, Tensor<TC> var2)
    {
        // Symmetric KL divergence between two Gaussians, element-wise
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        TC half = ops.FromDouble(0.5);
        var meanDiffSq = engine.TensorSquare(engine.TensorSubtract(mean1, mean2));
        var kl12 = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorAdd(engine.TensorAdd(
            engine.TensorLog(engine.TensorDivide(var2, var1)), engine.TensorDivide(var1, var2)),
            engine.TensorDivide(meanDiffSq, var2)), ops.One), half);
        var kl21 = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorAdd(engine.TensorAdd(
            engine.TensorLog(engine.TensorDivide(var1, var2)), engine.TensorDivide(var2, var1)),
            engine.TensorDivide(meanDiffSq, var1)), ops.One), half);
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
