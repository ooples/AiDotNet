using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// RReliefF algorithm for regression problems.
/// </summary>
/// <remarks>
/// <para>
/// RReliefF extends ReliefF to handle continuous targets (regression), following Robnik-Sikonja and
/// Kononenko (1997): over the k nearest neighbours of each sampled instance it accumulates the probability of a
/// different target (N_dC), of a different feature value (N_dA) and of both (N_dC&amp;dA), every neighbour having the
/// same influence 1/k, and weights each feature by
/// <c>W = N_dC&amp;dA / N_dC - (N_dA - N_dC&amp;dA) / (m - N_dC)</c>.
/// </para>
/// <para><b>For Beginners:</b> Standard Relief uses class labels (hit/miss).
/// RReliefF works with numeric targets by weighting neighbors based on
/// how similar their target values are to the sample's target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private Vector<T>? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public Vector<T>? FeatureWeights => _featureWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int nIterations = -1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RReliefF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int m = _nIterations > 0 ? Math.Min(_nIterations, n) : n;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        T infinity = NumOps.FromDouble(double.PositiveInfinity);

        // Features and target scaled to [0, 1] by their range (a constant one never differs)
        var scaled = FeatureSelectionTensorOps.RangeScaled(FilterStatisticTensorOps.AsTensor(data));
        var y = engine.Reshape(FeatureSelectionTensorOps.RangeScaled(
            engine.Reshape(FilterStatisticTensorOps.AsTensor(target), new[] { n, 1 })), new[] { n });

        // Sample m instances, in the same random order as before
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(m).ToArray();
        int sampled = sampleIndices.Length;
        var sampledRows = FeatureSelectionTensorOps.Rows(scaled, sampleIndices);
        var sampledY = FeatureSelectionTensorOps.Entries(y, sampleIndices);

        // k nearest neighbours by the sum of feature differences, the instance itself excluded
        var distance = FeatureSelectionTensorOps.KeepOrFill(
            engine.TensorCDist(sampledRows, scaled, 1.0),
            FilterStatisticTensorOps.OneMinus(FeatureSelectionTensorOps.SelfMask<T>(sampleIndices, n)), infinity);
        var neighbours = FeatureSelectionTensorOps.NearestPerRow(distance, _nNeighbors);

        // every one of an instance's neighbours has the same influence d = 1 / (number of neighbours)
        var neighbourCount = new Tensor<T>(new[] { sampled });
        foreach (var (_, found) in neighbours)
            neighbourCount = engine.TensorAdd(neighbourCount, found);

        // N_dC = sum d diff(target), N_dA = sum d diff(A), N_dC&dA = sum d diff(target) diff(A)
        var nD = new Tensor<T>(new[] { 1 });
        var nDa = new Tensor<T>(new[] { p });
        var nDcDa = new Tensor<T>(new[] { p });
        foreach (var (index, found) in neighbours)
        {
            var influence = FilterStatisticTensorOps.GuardedDivide(found, neighbourCount, 0.0, NumOps.Zero);
            var yDiff = engine.TensorAbs(engine.TensorSubtract(sampledY, FeatureSelectionTensorOps.Entries(y, index)));
            var featureDiff = FeatureSelectionTensorOps.NeighbourDifference(sampledRows, scaled, index);
            nD = engine.TensorAdd(nD, engine.ReduceSum(engine.TensorMultiply(influence, yDiff), new[] { 0 }, true));
            nDa = engine.TensorAdd(nDa, FeatureSelectionTensorOps.WeightedRowSum(influence, featureDiff));
            nDcDa = engine.TensorAdd(nDcDa, FeatureSelectionTensorOps.WeightedRowSum(
                engine.TensorMultiply(influence, yDiff), featureDiff));
        }

        // W[A] = N_dC&dA / N_dC - (N_dA - N_dC&dA) / (m - N_dC)  (Robnik-Sikonja and Kononenko 1997)
        var nDGrid = engine.TensorBroadcastTo(nD, new[] { p });
        var different = FilterStatisticTensorOps.GuardedDivide(nDcDa, nDGrid, 1e-10, NumOps.Zero);
        var restMass = engine.TensorAddScalar(engine.TensorNegate(nDGrid), NumOps.FromDouble(sampled));
        var same = FilterStatisticTensorOps.GuardedDivide(engine.TensorSubtract(nDa, nDcDa), restMass, 1e-10, NumOps.Zero);
        var weights = engine.TensorSubtract(different, same);
        _featureWeights = weights.ToVector();

        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(weights, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("RReliefF has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RReliefF does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RReliefF has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
