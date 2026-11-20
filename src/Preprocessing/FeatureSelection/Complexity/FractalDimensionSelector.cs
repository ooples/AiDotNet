using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Fractal Dimension based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their estimated fractal dimension using
/// the Higuchi algorithm, measuring signal complexity.
/// </para>
/// <para><b>For Beginners:</b> Fractal dimension measures how "rough" or complex
/// a signal is. A straight line has dimension 1, a completely random signal
/// approaches dimension 2. Complex, irregular patterns have higher fractal
/// dimensions. This is useful for finding features with interesting structure.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FractalDimensionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _kMax;

    private Vector<T>? _fractalDimensions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int KMax => _kMax;
    public Vector<T>? FractalDimensions => _fractalDimensions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FractalDimensionSelector(
        int nFeaturesToSelect = 10,
        int kMax = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (kMax < 2)
            throw new ArgumentException("kMax must be at least 2.", nameof(kMax));

        _nFeaturesToSelect = nFeaturesToSelect;
        _kMax = kMax;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < _kMax + 1)
            throw new ArgumentException($"Need at least {_kMax + 1} samples for kMax = {_kMax}.");

        var X = Tensor<T>.FromMatrix(data);
        // Half and BFloat16 run in float and the dimensions are cast back once: a curve length (7000 steps of 20
        // give L(1) = 139980) passes Half's 65504, and log(inf) entered the regression
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        if (PreprocessingTensorOps.IsNarrowFloat<T>())
        {
            var wideDimensions = ComputeHiguchiFD(X.Cast<float>(), n, p, _kMax);
            _fractalDimensions = wideDimensions.Cast<T>().ToVector();
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(wideDimensions, numToSelect);
        }
        else
        {
            var dimensions = ComputeHiguchiFD(X, n, p, _kMax);
            _fractalDimensions = dimensions.ToVector();
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(dimensions, numToSelect);
        }

        IsFitted = true;
    }

    private static Tensor<TC> ComputeHiguchiFD<TC>(Tensor<TC> X, int n, int p, int kMax)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        var curveLengths = new List<Tensor<TC>>();
        var scales = new List<int>();

        for (int k = 1; k <= kMax; k++)
        {
            Tensor<double>? lmSum = null;
            int lmCount = 0;

            for (int m = 1; m <= k; m++)
            {
                // Compute L_m(k) for every feature: the curve x[m-1], x[m-1+k], ... of numSegments steps
                int numSegments = (n - m) / k;
                if (numSegments < 1) continue;

                int start = m - 1;
                var rows = Enumerable.Range(0, numSegments + 1).Select(i => start + i * k).ToArray();
                var points = engine.TensorIndexSelect(X, new Tensor<int>(rows, new[] { rows.Length }), 0);
                var steps = engine.TensorAbs(engine.TensorSubtract(
                    engine.TensorSlice(points, new[] { 1, 0 }, new[] { numSegments, p }),
                    engine.TensorSlice(points, new[] { 0, 0 }, new[] { numSegments, p })));
                // Summed and normalized in double: a Half sum of the steps stalls once it passes 2048 times their
                // spacing, and the normalizer k^2 numSegments overflows Half's 65504 maximum
                var lm = engine.TensorMultiplyScalar(engine.ReduceSum(steps.Cast<double>(), new[] { 0 }, false),
                    (n - 1.0) / ((double)k * k * numSegments));
                lmSum = lmSum is null ? lm : engine.TensorAdd(lmSum, lm);
                lmCount++;
            }

            if (lmSum is not null)
            {
                curveLengths.Add(engine.TensorDivideScalar(lmSum, (double)lmCount).Cast<TC>());
                scales.Add(k);
            }
        }

        var defaultDimension = PreprocessingTensorOps.Filled(new[] { p }, ops.FromDouble(1.5));
        if (scales.Count == 0)
            return defaultDimension;

        // Per feature, only the scales whose average curve length is positive enter the fit
        int count = scales.Count;
        var shape = new[] { count, p };
        var avgL = engine.TensorStack(curveLengths.ToArray(), 0);
        var valid = engine.TensorGreaterThan(avgL, ops.Zero);
        var zeros = new Tensor<TC>(shape);
        var logL = engine.TensorWhere(valid, engine.TensorLog(avgL), zeros);
        var logK = engine.TensorBroadcastTo(engine.Reshape(
            engine.TensorLog(new Tensor<int>(scales.ToArray(), new[] { count }).Cast<TC>()), new[] { count, 1 }), shape);
        var maskedLogK = engine.TensorMultiply(logK, valid);

        // Linear regression of log L on log k: FD = -slope
        var pointCount = engine.ReduceSum(valid, new[] { 0 }, true);
        var meanX = engine.TensorDivide(engine.ReduceSum(maskedLogK, new[] { 0 }, true), pointCount);
        var meanY = engine.TensorDivide(engine.ReduceSum(logL, new[] { 0 }, true), pointCount);
        var dx = engine.TensorMultiply(engine.TensorSubtract(logK, engine.TensorBroadcastTo(meanX, shape)), valid);
        var dy = engine.TensorMultiply(engine.TensorSubtract(logL, engine.TensorBroadcastTo(meanY, shape)), valid);
        var numerator = engine.ReduceSum(engine.TensorMultiply(dx, dy), new[] { 0 }, false);
        var denominator = engine.ReduceSum(engine.TensorMultiply(dx, dx), new[] { 0 }, false);

        // Clamp to valid range [1, 2]; fewer than 2 points or a flat log k gives the default dimension
        var fd = engine.TensorClamp(
            engine.TensorNegate(engine.TensorDivide(numerator, denominator)), ops.One, ops.FromDouble(2));
        var fitted = engine.TensorMultiply(
            PreprocessingTensorOps.AtLeast(engine.Reshape(pointCount, new[] { p }), ops.FromDouble(2)),
            PreprocessingTensorOps.Not(engine.TensorLessThan(denominator, PreprocessingTensorOps.Tolerance<TC>(1e-10))));
        return engine.TensorWhere(fitted, fd, defaultDimension);
    }
    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FractalDimensionSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FractalDimensionSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FractalDimensionSelector has not been fitted.");

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
