using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Bayesian Feature Selection using posterior probability.
/// </summary>
/// <remarks>
/// <para>
/// Uses Bayesian inference to compute the posterior probability that each
/// feature is relevant to the target. Features with high posterior probability
/// are selected.
/// </para>
/// <para><b>For Beginners:</b> Bayesian methods treat feature relevance as a
/// probability question. Instead of saying "feature X is important" or "not important",
/// we say "there's a 90% chance feature X is important." This gives us uncertainty
/// estimates and allows us to incorporate prior knowledge about which features
/// might be relevant.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BayesianFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _priorProbability;
    private readonly double _threshold;

    private Vector<T>? _posteriorProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double PriorProbability => _priorProbability;
    public Vector<T>? PosteriorProbabilities => _posteriorProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BayesianFeatureSelector(
        int nFeaturesToSelect = 10,
        double priorProbability = 0.5,
        double threshold = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (priorProbability <= 0 || priorProbability >= 1)
            throw new ArgumentException("Prior probability must be between 0 and 1 exclusive.", nameof(priorProbability));
        if (threshold < 0 || threshold > 1)
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be in range [0, 1].");

        _nFeaturesToSelect = nFeaturesToSelect;
        _priorProbability = priorProbability;
        _threshold = threshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BayesianFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Class masks as [1, n] rows so the per-class sums are one matmul each.
        var positive = engine.TensorGreaterThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5));
        var negative = engine.TensorAddScalar(engine.TensorNegate(positive), NumOps.One);
        var positiveRow = engine.Reshape(positive, new[] { 1, n });
        var negativeRow = engine.Reshape(negative, new[] { 1, n });
        int countPos = PreprocessingTensorOps.CountNonZero(positive);
        int countNeg = n - countPos;

        if (countPos < 2 || countNeg < 2)
        {
            _posteriorProbabilities = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(_priorProbability)).ToVector();
        }
        else
        {
            var squares = engine.TensorSquare(X);
            var sumPos = engine.Reshape(engine.TensorMatMul(positiveRow, X), new[] { p });
            var sumNeg = engine.Reshape(engine.TensorMatMul(negativeRow, X), new[] { p });
            var sumSqPos = engine.Reshape(engine.TensorMatMul(positiveRow, squares), new[] { p });
            var sumSqNeg = engine.Reshape(engine.TensorMatMul(negativeRow, squares), new[] { p });

            T eps = NumOps.FromDouble(1e-10);
            var meanPos = engine.TensorDivideScalar(sumPos, NumOps.FromDouble(countPos));
            var meanNeg = engine.TensorDivideScalar(sumNeg, NumOps.FromDouble(countNeg));
            var varPos = engine.TensorAddScalar(
                engine.TensorSubtract(engine.TensorDivideScalar(sumSqPos, NumOps.FromDouble(countPos)), engine.TensorSquare(meanPos)), eps);
            var varNeg = engine.TensorAddScalar(
                engine.TensorSubtract(engine.TensorDivideScalar(sumSqNeg, NumOps.FromDouble(countNeg)), engine.TensorSquare(meanNeg)), eps);

            // Bayes factor using a t-test approximation
            var pooledVar = engine.TensorDivideScalar(
                engine.TensorAdd(
                    engine.TensorMultiplyScalar(varPos, NumOps.FromDouble(countPos - 1)),
                    engine.TensorMultiplyScalar(varNeg, NumOps.FromDouble(countNeg - 1))),
                NumOps.FromDouble(n - 2));
            var standardError = engine.TensorAddScalar(
                engine.TensorSqrt(engine.TensorMultiplyScalar(pooledVar, NumOps.FromDouble(1.0 / countPos + 1.0 / countNeg))), eps);
            var tStatistic = engine.TensorDivide(engine.TensorAbs(engine.TensorSubtract(meanPos, meanNeg)), standardError);

            // t-statistic to likelihood ratio approximation
            var logLikelihoodRatio = engine.TensorMin(
                engine.TensorDivideScalar(engine.TensorSquare(tStatistic), NumOps.FromDouble(2.0)), NumOps.FromDouble(50.0));
            var likelihoodRatio = engine.TensorExp(logLikelihoodRatio);

            // Posterior using Bayes' rule
            var posteriorOdds = engine.TensorMultiplyScalar(
                likelihoodRatio, NumOps.FromDouble(_priorProbability / (1 - _priorProbability)));
            _posteriorProbabilities = engine.TensorDivide(posteriorOdds, engine.TensorAddScalar(posteriorOdds, NumOps.One)).ToVector();
        }

        // Select features that exceed threshold, up to nFeaturesToSelect
        _selectedIndices = PreprocessingTensorOps.TopKAtLeastSorted(
            Tensor<T>.FromVector(_posteriorProbabilities), NumOps.FromDouble(_threshold), _nFeaturesToSelect);

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
            throw new InvalidOperationException("BayesianFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BayesianFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianFeatureSelector has not been fitted.");

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
