using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Spike-and-Slab Feature Selection using Bayesian variable selection.
/// </summary>
/// <remarks>
/// <para>
/// Implements the spike-and-slab prior model where each feature coefficient
/// has a mixture prior: a "spike" (concentrated at zero) and a "slab" (diffuse).
/// Features with high probability of being in the slab are selected.
/// </para>
/// <para><b>For Beginners:</b> Imagine a spike-and-slab as two possibilities for
/// each feature: either it has no effect (spike at zero) or it has some effect
/// (spread out slab). The method estimates which possibility is more likely for
/// each feature. If the slab is more likely, the feature is important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SpikeAndSlabSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _slabVariance;
    private readonly double _spikeVariance;
    private readonly double _priorInclusionProbability;
    private readonly int _nIterations;

    private Vector<T>? _inclusionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? InclusionProbabilities => _inclusionProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SpikeAndSlabSelector(
        int nFeaturesToSelect = 10,
        double slabVariance = 1.0,
        double spikeVariance = 0.001,
        double priorInclusionProbability = 0.5,
        int nIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (slabVariance <= 0)
            throw new ArgumentException("Slab variance must be positive.", nameof(slabVariance));
        if (spikeVariance <= 0)
            throw new ArgumentException("Spike variance must be positive.", nameof(spikeVariance));
        if (priorInclusionProbability <= 0 || priorInclusionProbability >= 1)
            throw new ArgumentException("Prior inclusion probability must be between 0 and 1 exclusive.", nameof(priorInclusionProbability));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _slabVariance = slabVariance;
        _spikeVariance = spikeVariance;
        _priorInclusionProbability = priorInclusionProbability;
        _nIterations = nIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SpikeAndSlabSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Half and BFloat16 are evaluated in float and the probabilities cast back once: the squared centred targets
        // (100 .. 800 gives 245,000) and the fit terms pass Half's 65504, and the non-finite Bayes factor fell back
        // to the prior
        var gamma = PreprocessingTensorOps.IsNarrowFloat<T>()
            ? ComputeInclusionProbabilities(X.Cast<float>(), y.Cast<float>(),
                _slabVariance, _spikeVariance, _priorInclusionProbability).Cast<T>()
            : ComputeInclusionProbabilities(X, y, _slabVariance, _spikeVariance, _priorInclusionProbability);
        _inclusionProbabilities = gamma.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(gamma, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Posterior inclusion probability of every column of <paramref name="X"/> ([n, p]) for target
    /// <paramref name="y"/> ([n]) under the spike-and-slab Bayes factor approximation.
    /// </summary>
    private static Tensor<TC> ComputeInclusionProbabilities<TC>(
        Tensor<TC> X, Tensor<TC> y, double slabVariance, double spikeVariance, double priorInclusionProbability)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        int n = X.Shape[0];
        int p = X.Shape[1];
        TC eps = PreprocessingTensorOps.Tolerance<TC>(1e-10);
        var prior = PreprocessingTensorOps.Filled(new[] { p }, ops.FromDouble(priorInclusionProbability));

        // Residual variance estimate
        TC residualVariance = ops.Add(
            ops.Divide(engine.TensorSumOfSquares(engine.TensorSubtractScalar(y, engine.TensorMean(y))), ops.FromDouble(n - 1)),
            eps);

        // Marginal likelihood ratio per feature. Every sweep of the sampler sees the same data, so each
        // iteration yields the same inclusion probabilities and their average over iterations is this value.
        // ||x_j||^2 overflows Half (65504) once a value passes 256, so every column is divided by its largest
        // magnitude s first: with u = x / s, ||x_j||^2 = s^2 ||u_j||^2, and the terms master formed from it are
        // fit = 0.5 betaHat^2 ||x_j||^2 = 0.5 (x_j'y / ||x_j||)^2 = 0.5 (u_j'y / ||u_j||)^2 and
        // residualVariance / ||x_j||^2 = ((residualVariance / s) / s) / ||u_j||^2, none of which squares s.
        var scale = PreprocessingTensorOps.ColumnMax(engine.TensorAbs(X));
        scale = engine.TensorWhere(engine.TensorGreaterThan(scale, ops.Zero), scale, PreprocessingTensorOps.Filled(new[] { p }, ops.One));
        var unit = engine.TensorDivide(X, engine.TensorBroadcastTo(engine.Reshape(scale, new[] { 1, p }), new[] { n, p }));
        var unitNormSquared = engine.ReduceSum(engine.TensorSquare(unit), new[] { 0 }, false);
        var unitNorm = engine.TensorSqrt(unitNormSquared);
        var uy = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(unit), engine.Reshape(y, new[] { n, 1 })), new[] { p });
        var fitTerm = engine.TensorMultiplyScalar(engine.TensorSquare(engine.TensorDivide(uy, unitNorm)), ops.FromDouble(0.5));
        var varianceRatio = engine.TensorDivide(
            engine.TensorDivide(engine.TensorDivide(PreprocessingTensorOps.Filled(new[] { p }, residualVariance), scale), scale),
            unitNormSquared);
        // ||x_j|| = s ||u_j||, compared with sqrt(1e-10) where master compared ||x_j||^2 with 1e-10
        var xjNorm = engine.TensorMultiply(scale, unitNorm);
        TC minusHalf = ops.FromDouble(-0.5);

        // Bayes factor approximation
        var logBFSlab = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorLog(engine.TensorAddScalar(varianceRatio, ops.FromDouble(slabVariance))), minusHalf),
            engine.TensorDivideScalar(fitTerm, ops.Add(residualVariance, eps)));
        var logBFSpike = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorLog(engine.TensorAddScalar(varianceRatio, ops.FromDouble(spikeVariance))), minusHalf),
            engine.TensorDivideScalar(fitTerm,
                ops.Add(ops.Multiply(residualVariance, ops.FromDouble(spikeVariance / slabVariance)), eps)));
        var logBF = engine.TensorSubtract(logBFSlab, logBFSpike);

        // Posterior inclusion odds / (1 + odds) with odds = prior odds * exp(min(logBF, 50)), evaluated as
        // sigmoid(min(logBF, 50) + log prior odds): the ratio form overflows to inf / inf = NaN once the odds pass the
        // largest finite T (e^11.1 in Half), while the sigmoid saturates at 1.
        TC logPriorOdds = ops.FromDouble(Math.Log(priorInclusionProbability / (1 - priorInclusionProbability)));
        var gamma = engine.TensorSigmoid(
            engine.TensorAddScalar(engine.TensorMin(logBF, ops.FromDouble(50.0)), logPriorOdds));

        // Degenerate columns (||x_j||^2 < 1e-10) and non-finite Bayes factors keep the prior inclusion probability.
        gamma = engine.TensorWhere(engine.TensorIsFinite(logBF), gamma, prior);
        gamma = engine.TensorWhere(engine.TensorLessThan(xjNorm, ops.Sqrt(eps)), prior, gamma);
        return gamma;
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
            throw new InvalidOperationException("SpikeAndSlabSelector has not been fitted.");
        if (data.Columns != _nInputFeatures)
            throw new ArgumentException(
                $"Expected {_nInputFeatures} columns but got {data.Columns}.", nameof(data));

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpikeAndSlabSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpikeAndSlabSelector has not been fitted.");

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
