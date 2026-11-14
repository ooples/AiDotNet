using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Association;

/// <summary>
/// Apriori-based Association Rule Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses association rule mining concepts to find features that frequently
/// co-occur with the target class, selecting features with high support
/// and confidence.
/// </para>
/// <para><b>For Beginners:</b> Association rules find patterns like "people who
/// buy bread often buy butter." We apply this to features: features that
/// "associate" strongly with the target class (high confidence) and appear
/// often (high support) are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AprioriBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minSupport;
    private readonly double _minConfidence;
    private readonly int _nBins;

    private Vector<T>? _associationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? AssociationScores => _associationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AprioriBasedSelector(
        int nFeaturesToSelect = 10,
        double minSupport = 0.1,
        double minConfidence = 0.5,
        int nBins = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minSupport = minSupport;
        _minConfidence = minConfidence;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AprioriBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var positiveRows = PreprocessingTensorOps.MaskPositions(
            engine.TensorGreaterThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5))); // Binarize
        int nPositive = positiveRows.Length;
        double targetSupport = (double)nPositive / n;

        // Discretize features: bin = min(floor((x - min) / (max - min + 1e-10) * nBins), nBins - 1), as the cell
        // j * nBins + bin of a (feature, bin) histogram. Half and BFloat16 are binned in float: max - min of finite
        // Half values overflows past 65504, and 1e-10 (raised to 2^-14 in Half) would merge small-spread bins.
        int cells = p * _nBins;
        var cell = PreprocessingTensorOps.IsNarrowFloat<T>()
            ? HistogramCells(X.Cast<float>(), _nBins)
            : HistogramCells(X, _nBins);

        // Per (feature, bin): transactions with the value and those that are also positive, counted in int by a
        // histogram over the [n, p] cells (no [n, p * nBins] one-hot; a Half sum of 0/1 entries stops at 2048)
        var featureCounts = CellCounts(engine, engine.Reshape(cell, new[] { n * p }), cells).Cast<double>();
        var jointCounts = nPositive == 0
            ? new Tensor<double>(new[] { cells })
            : CellCounts(engine, engine.Reshape(engine.TensorIndexSelect(
                cell, new Tensor<int>(positiveRows, new[] { nPositive }), 0), new[] { nPositive * p }), cells).Cast<double>();

        // Support, confidence and lift are ratios of the int counts, formed in double as in master: the counts
        // themselves pass Half's 65504 maximum (70,000 rows), which would zero the support
        var support = engine.TensorDivideScalar(jointCounts, (double)n);
        var confidence = engine.TensorDivide(jointCounts, featureCounts);
        var lift = engine.TensorDivideScalar(confidence, targetSupport + 1e-10);

        // Score combines support, confidence, and lift for the rules that clear both minimums
        var score = engine.TensorMultiply(engine.TensorMultiply(support, confidence), lift);
        var qualifies = engine.TensorMultiply(
            engine.TensorMultiply(
                engine.TensorGreaterThan(featureCounts, 0.0),
                PreprocessingTensorOps.AtLeast(support, _minSupport)),
            PreprocessingTensorOps.AtLeast(confidence, _minConfidence));
        var masked = engine.TensorWhere(qualifies, score, new Tensor<double>(new[] { cells }));
        var scores = engine.TensorMax(
            engine.ReduceMax(engine.Reshape(masked, new[] { p, _nBins }), new[] { 1 }, false), 0.0);
        _associationScores = scores.Cast<T>().ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// The histogram cell <c>j * bins + bin</c> of every entry of <paramref name="x"/> ([n, p]), with
    /// <c>bin = min(floor((x - min) / (max - min + 1e-10) * bins), bins - 1)</c> per column (a constant column bins at
    /// 0, dividing by 1 rather than 1e-10). A NaN entry is in no bin (master's <c>(int)NaN</c> matched none) and takes
    /// the overflow cell <c>p * bins</c>.
    /// </summary>
    private static Tensor<int> HistogramCells<TC>(Tensor<TC> x, int bins)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var min = engine.Reshape(PreprocessingTensorOps.ColumnMin(x), new[] { 1, p });
        var spread = engine.TensorSubtract(engine.Reshape(PreprocessingTensorOps.ColumnMax(x), new[] { 1, p }), min);
        var range = engine.TensorWhere(engine.TensorGreaterThan(spread, ops.Zero),
            engine.TensorAddScalar(spread, PreprocessingTensorOps.Tolerance<TC>(1e-10)),
            PreprocessingTensorOps.Filled(new[] { 1, p }, ops.One));
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(x, min), range), ops.FromDouble(bins));
        var bin = engine.TensorMin(engine.TensorFloor(scaled), ops.FromDouble(bins - 1));
        var number = engine.TensorEquals(scaled, scaled);
        var isNumber = number.Cast<int>();
        var offsets = new Tensor<int>(Enumerable.Range(0, p).Select(j => j * bins).ToArray(), new[] { 1, p });
        var cell = engine.TensorAdd(
            engine.TensorWhere(number, bin, new Tensor<TC>(new[] { n, p })).Cast<int>(),
            engine.TensorBroadcastTo(offsets, new[] { n, p }));
        return engine.TensorWhere(isNumber, cell, PreprocessingTensorOps.Filled(new[] { n, p }, p * bins));
    }

    /// <summary>Int counts of the cells 0..<paramref name="cells"/>-1 (the overflow cell is dropped).</summary>
    private static Tensor<int> CellCounts(AiDotNet.Tensors.Engines.IEngine engine, Tensor<int> cell, int cells) =>
        engine.TensorNarrow(engine.TensorBinCount(cell, cells + 1), 0, 0, cells);

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
            throw new InvalidOperationException("AprioriBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AprioriBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AprioriBasedSelector has not been fitted.");

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
