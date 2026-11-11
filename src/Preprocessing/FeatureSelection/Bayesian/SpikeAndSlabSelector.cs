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
public class SpikeAndSlabSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _slabVariance;
    private readonly double _spikeVariance;
    private readonly double _priorInclusionProbability;
    private readonly int _nIterations;

    private Vector<T>? _inclusionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? InclusionProbabilities => _inclusionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SpikeAndSlabSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        T eps = NumOps.FromDouble(1e-10);
        var prior = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(_priorInclusionProbability));

        // Residual variance estimate
        T residualVariance = NumOps.Add(
            NumOps.Divide(engine.TensorSumOfSquares(engine.TensorSubtractScalar(y, engine.TensorMean(y))), NumOps.FromDouble(n - 1)),
            eps);

        // Marginal likelihood ratio per feature. Every sweep of the sampler sees the same data, so each
        // iteration yields the same inclusion probabilities and their average over iterations is this value.
        var xjNorm = engine.ReduceSum(engine.TensorSquare(X), new[] { 0 }, false);
        var xjy = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(X), engine.Reshape(y, new[] { n, 1 })), new[] { p });
        var betaHat = engine.TensorDivide(xjy, xjNorm);
        var fitTerm = engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorSquare(betaHat), xjNorm), NumOps.FromDouble(0.5));
        var varianceRatio = engine.TensorDivide(PreprocessingTensorOps.Filled(new[] { p }, residualVariance), xjNorm);
        T minusHalf = NumOps.FromDouble(-0.5);

        // Bayes factor approximation
        var logBFSlab = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorLog(engine.TensorAddScalar(varianceRatio, NumOps.FromDouble(_slabVariance))), minusHalf),
            engine.TensorDivideScalar(fitTerm, NumOps.Add(residualVariance, eps)));
        var logBFSpike = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorLog(engine.TensorAddScalar(varianceRatio, NumOps.FromDouble(_spikeVariance))), minusHalf),
            engine.TensorDivideScalar(fitTerm,
                NumOps.Add(NumOps.Multiply(residualVariance, NumOps.FromDouble(_spikeVariance / _slabVariance)), eps)));
        var logBF = engine.TensorSubtract(logBFSlab, logBFSpike);

        double priorOdds = _priorInclusionProbability / (1 - _priorInclusionProbability);
        var posteriorOdds = engine.TensorMultiplyScalar(
            engine.TensorExp(engine.TensorMin(logBF, NumOps.FromDouble(50.0))), NumOps.FromDouble(priorOdds));
        var gamma = engine.TensorDivide(posteriorOdds, engine.TensorAddScalar(posteriorOdds, NumOps.One));

        // Degenerate columns (xjNorm < 1e-10) and non-finite Bayes factors keep the prior inclusion probability.
        gamma = engine.TensorWhere(engine.TensorIsFinite(logBF), gamma, prior);
        gamma = engine.TensorWhere(engine.TensorLessThan(xjNorm, eps), prior, gamma);
        _inclusionProbabilities = gamma.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(gamma, numToSelect);

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpikeAndSlabSelector has not been fitted.");
        if (data.Columns != _nInputFeatures)
            throw new ArgumentException(
                $"Expected {_nInputFeatures} columns but got {data.Columns}.", nameof(data));

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpikeAndSlabSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpikeAndSlabSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
