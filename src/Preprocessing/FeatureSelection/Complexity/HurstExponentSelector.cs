using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Hurst Exponent based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Hurst exponent, which measures
/// long-range dependence and self-similarity in time series.
/// </para>
/// <para><b>For Beginners:</b> The Hurst exponent H indicates trend behavior:
/// H &gt; 0.5 means trending (persistent) - ups followed by ups, downs by downs;
/// H &lt; 0.5 means mean-reverting (anti-persistent) - ups followed by downs;
/// H = 0.5 means random walk (no memory). Financial time series with H ≠ 0.5
/// may contain exploitable patterns.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HurstExponentSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferTrending;

    private Vector<T>? _hurstValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferTrending => _preferTrending;
    public Vector<T>? HurstValues => _hurstValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public HurstExponentSelector(
        int nFeaturesToSelect = 10,
        bool preferTrending = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferTrending = preferTrending;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 20)
            throw new ArgumentException("Need at least 20 samples for Hurst exponent estimation.");

    }
@@NEW
        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var hurst = ComputeHurstRS(X, n, p);
        _hurstValues = hurst.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferTrending)
        {
            // Prefer features with H > 0.5 (trending/persistent)
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(hurst, numToSelect);
        }
        else
        {
            // Prefer features with H furthest from 0.5 (strong memory either way)
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(
                engine.TensorAbs(engine.TensorSubtractScalar(hurst, NumOps.FromDouble(0.5))), numToSelect);
        }

        IsFitted = true;
    }
    }

    private Tensor<T> ComputeHurstRS(Tensor<T> X, int n, int p)
    {
        var engine = Engine;

        // Use rescaled range (R/S) method, for every feature at once
        var lagSizes = new List<int>();
        var rsValues = new List<Tensor<T>>();
        var validValues = new List<Tensor<T>>();
        T stdFloor = NumOps.FromDouble(1e-10);
        T zero = NumOps.Zero;

        // Choose lag sizes from 10 to n/2
        for (int lag = 10; lag <= n / 2; lag = (int)(lag * 1.5))
        {
            // Divide series into subseries: [numSubseries, lag, p]
            int numSubseries = n / lag;
            var subseries = engine.Reshape(
                engine.TensorSlice(X, new[] { 0, 0 }, new[] { numSubseries * lag, p }), new[] { numSubseries, lag, p });
            var deviations = engine.TensorSubtract(subseries, engine.TensorBroadcastTo(
                engine.ReduceMean(subseries, new[] { 1 }, true), new[] { numSubseries, lag, p }));

            // Range of the cumulative deviations
            var cumDev = engine.TensorCumSum(deviations, 1);
            var range = engine.TensorSubtract(
                engine.ReduceMax(cumDev, new[] { 1 }, false),
                engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(cumDev), new[] { 1 }, false)));

            // Standard deviation (population)
            var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(deviations), new[] { 1 }, false));

            // Average R/S over the subseries whose standard deviation is above 1e-10
            var usable = engine.TensorGreaterThan(std, stdFloor);
            var rs = engine.TensorWhere(usable, engine.TensorDivide(range, std), new Tensor<T>(new[] { numSubseries, p }));
            var usableCount = engine.ReduceSum(usable, new[] { 0 }, false);
            rsValues.Add(engine.TensorDivide(engine.ReduceSum(rs, new[] { 0 }, false), usableCount));
            validValues.Add(engine.TensorGreaterThan(usableCount, zero));
            lagSizes.Add(lag);
        }

        var randomWalk = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(0.5));
        if (lagSizes.Count == 0)
            return randomWalk;

        // Linear regression of log(R/S) vs log(lag): H = slope, over each feature's usable lags
        int count = lagSizes.Count;
        var valid = engine.TensorStack(validValues.ToArray(), 0);
        var logRS = engine.TensorWhere(valid, engine.TensorLog(engine.TensorStack(rsValues.ToArray(), 0)), new Tensor<T>(new[] { count, p }));
        var logLag = engine.TensorLog(new Tensor<int>(lagSizes.ToArray(), new[] { count }).Cast<T>());
        var slope = FeatureSelectionTensorOps.MaskedSlope(logLag, logRS, valid, 1e-10, out var fitted);

        // Clamp to valid range [0, 1]; too few lags or a flat log(lag) gives 0.5 (random walk)
        return engine.TensorWhere(fitted, engine.TensorClamp(slope, NumOps.Zero, NumOps.One), randomWalk);
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
            throw new InvalidOperationException("HurstExponentSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HurstExponentSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HurstExponentSelector has not been fitted.");

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
