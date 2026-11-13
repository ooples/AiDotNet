using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Diverse Ensemble Feature Selection using multiple diverse methods.
/// </summary>
/// <remarks>
/// <para>
/// Combines multiple feature selection methods with intentionally different
/// characteristics (filter, wrapper, embedded) to get robust rankings.
/// Features consistently ranked high across diverse methods are selected.
/// </para>
/// <para><b>For Beginners:</b> Different feature selection methods look at
/// features from different angles - some focus on correlation, others on
/// prediction power, others on regularization. By combining diverse perspectives,
/// we identify features that are truly important from multiple viewpoints.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DiverseEnsemble<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int? _randomState;

    private Vector<T>? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? AggregatedScores => _aggregatedScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DiverseEnsemble(
        int nFeaturesToSelect = 10,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DiverseEnsemble requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var allScores = new List<Tensor<T>>();

        // Method 1: Correlation-based
        allScores.Add(engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, y)));

        // Method 2: Mutual Information-based
        allScores.Add(ComputeMutualInformationScores(X, y, n, p));

        // Method 3: F-statistic (ANOVA)
        allScores.Add(ComputeFScores(X, target, n, p));

        // Method 4: Variance-based (population variance)
        allScores.Add(engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(X,
            engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { n, p }))), new[] { 0 }, false));

        // Method 5: Distance correlation
        allScores.Add(ComputeDistanceCorrelationScores(X, y, n, random));

        // Aggregate scores (normalize each method to [0, 1], 0 when its range is not above 1e-10, and average)
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var aggregatedScores = new Tensor<T>(new[] { p });

        foreach (var scores in allScores)
        {
            var shifted = engine.TensorSubtractScalar(scores, engine.TensorMinValue(scores));
            T range = engine.TensorMaxValue(shifted);
            if (PreprocessingTensorOps.IsAbove(range, floor))
                aggregatedScores = engine.TensorAdd(aggregatedScores, engine.TensorDivideScalar(shifted, range));
        }

        // Average across methods
        aggregatedScores = engine.TensorDivideScalar(aggregatedScores, NumOps.FromDouble(allScores.Count));
        _aggregatedScores = aggregatedScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(aggregatedScores, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeMutualInformationScores(Tensor<T> X, Tensor<T> y, int n, int p)
    {
        // 10 range-scaled bins per feature and for the target; MI in nats over the non-empty joint cells
        int nBins = 10;
        var xBins = PreprocessingTensorOps.RangeScaledBinOneHot(X, nBins);
        var yBins = PreprocessingTensorOps.RangeScaledBinOneHot(Engine.Reshape(y, new[] { n, 1 }), nBins);
        var mi = PreprocessingTensorOps.HistogramMutualInformation(xBins, yBins, nBins, bits: false, logEpsilon: 0, clampAtZero: false);
        return Engine.Reshape(mi, new[] { p });
    }

    private Tensor<T> ComputeFScores(Tensor<T> X, Vector<T> target, int n, int p)
    {
        var engine = Engine;

        // Group by target class
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        if (k < 2)
            return new Tensor<T>(new[] { p });

        var membership = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
        // Class sizes and means counted exactly (a Half sum of the 0/1 membership stops at 2048)
        var classMeans = FeatureSelectionTensorOps.GroupMeans(membership, X, out var counts);
        var grandMean = engine.ReduceMean(X, new[] { 0 }, true);

        // Between-class and within-class sums of squares
        var ssb = engine.ReduceSum(engine.TensorMultiply(
            engine.TensorSquare(engine.TensorSubtract(classMeans, engine.TensorBroadcastTo(grandMean, new[] { k, p }))),
            engine.TensorBroadcastTo(engine.Reshape(counts, new[] { k, 1 }), new[] { k, p })), new[] { 0 }, false);
        var rowMeans = engine.TensorIndexSelect(classMeans, new Tensor<int>(classIndex, new[] { n }), 0);
        var ssw = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(X, rowMeans)), new[] { 0 }, false);

        int dfb = k - 1;
        int dfw = n - k;
        if (dfw <= 0)
            return new Tensor<T>(new[] { p });

        var usable = engine.TensorGreaterThan(ssw, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var safeSsw = engine.TensorWhere(usable, ssw, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        return engine.TensorWhere(usable, engine.TensorDivide(
            engine.TensorDivideScalar(ssb, NumOps.FromDouble(dfb)), engine.TensorDivideScalar(safeSsw, NumOps.FromDouble(dfw))),
            new Tensor<T>(new[] { p }));
    }

    private Tensor<T> ComputeDistanceCorrelationScores(Tensor<T> X, Tensor<T> y, int n, Random random)
    {
        var engine = Engine;

        // Use a sample for computational efficiency
        int sampleSize = Math.Min(100, n);
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(sampleSize).ToArray();
        var rows = new Tensor<int>(sampleIndices, new[] { sampleSize });

        // Distance correlation dCor = sqrt(dCov^2 / sqrt(dVar^2(x) dVar^2(y))), 0 when a distance variance is 0
        var dcov2 = FeatureSelectionTensorOps.SquaredDistanceCovariances(
            engine.TensorIndexSelect(X, rows, 0), engine.TensorIndexSelect(y, rows, 0), out var xVariances, out T yVariance);
        int p = dcov2.Length;
        // sqrt(a) sqrt(b) rather than sqrt(a b): the product of the two variances overflows Half's 65504 maximum
        var denominators = engine.TensorMultiplyScalar(engine.TensorSqrt(xVariances), NumOps.Sqrt(yVariance));
        var usable = engine.TensorGreaterThan(denominators, NumOps.Zero);
        var safe = engine.TensorWhere(usable, denominators, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        return engine.TensorWhere(usable,
            engine.TensorSqrt(engine.TensorMax(engine.TensorDivide(dcov2, safe), NumOps.Zero)), new Tensor<T>(new[] { p }));
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
            throw new InvalidOperationException("DiverseEnsemble has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DiverseEnsemble does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DiverseEnsemble has not been fitted.");

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
