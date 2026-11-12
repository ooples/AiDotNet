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
        var dimensions = ComputeHiguchiFD(X, n, p);
        _fractalDimensions = dimensions.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(dimensions, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeHiguchiFD(Tensor<T> X, int n, int p)
    {
        var engine = Engine;
        T nMinusOne = NumOps.FromDouble(n - 1.0);
        var curveLengths = new List<Tensor<T>>();
        var scales = new List<int>();

        for (int k = 1; k <= _kMax; k++)
        {
            Tensor<T>? lmSum = null;
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
                var lm = engine.TensorDivideScalar(
                    engine.TensorMultiplyScalar(engine.ReduceSum(steps, new[] { 0 }, false), nMinusOne),
                    PreprocessingTensorOps.CountAsScalar<T>(k * k * numSegments));
                lmSum = lmSum is null ? lm : engine.TensorAdd(lmSum, lm);
                lmCount++;
            }

            if (lmSum is not null)
            {
                curveLengths.Add(engine.TensorDivideScalar(lmSum, PreprocessingTensorOps.CountAsScalar<T>(lmCount)));
                scales.Add(k);
            }
        }

        var defaultDimension = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(1.5));
        if (scales.Count == 0)
            return defaultDimension;

        // Per feature, only the scales whose average curve length is positive enter the fit
        int count = scales.Count;
        var shape = new[] { count, p };
        var avgL = engine.TensorStack(curveLengths.ToArray(), 0);
        var valid = engine.TensorGreaterThan(avgL, NumOps.Zero);
        var zeros = new Tensor<T>(shape);
        var logL = engine.TensorWhere(valid, engine.TensorLog(avgL), zeros);
        var logK = engine.TensorBroadcastTo(engine.Reshape(
            engine.TensorLog(new Tensor<int>(scales.ToArray(), new[] { count }).Cast<T>()), new[] { count, 1 }), shape);
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
            engine.TensorNegate(engine.TensorDivide(numerator, denominator)), NumOps.One, NumOps.FromDouble(2));
        var fitted = engine.TensorMultiply(
            PreprocessingTensorOps.AtLeast(engine.Reshape(pointCount, new[] { p }), NumOps.FromDouble(2)),
            PreprocessingTensorOps.Not(engine.TensorLessThan(denominator, NumOps.FromDouble(1e-10))));
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
