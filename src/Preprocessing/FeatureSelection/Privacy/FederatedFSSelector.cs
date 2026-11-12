using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Privacy;

/// <summary>
/// Federated Feature Selection Selector.
/// </summary>
/// <remarks>
/// <para>
/// Simulates federated feature selection where data is partitioned across
/// multiple clients, and feature importance is computed locally then aggregated.
/// </para>
/// <para><b>For Beginners:</b> Federated learning keeps data on local devices/servers
/// (clients) and only shares model updates. This selector simulates that by:
/// 1) Splitting data into partitions (simulating different clients)
/// 2) Computing feature importance locally on each partition
/// 3) Aggregating scores across clients without sharing raw data
/// This provides privacy as raw data never leaves its partition.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FederatedFSSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClients;
    private readonly FederatedScoreAggregation _aggregationMethod;

    private Vector<T>? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NClients => _nClients;
    public FederatedScoreAggregation AggregationMethod => _aggregationMethod;
    public Vector<T>? AggregatedScores => _aggregatedScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FederatedFSSelector(
        int nFeaturesToSelect = 10,
        int nClients = 5,
        FederatedScoreAggregation aggregationMethod = FederatedScoreAggregation.WeightedAverage,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nClients < 2)
            throw new ArgumentException("Number of clients must be at least 2.", nameof(nClients));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClients = nClients;
        _aggregationMethod = aggregationMethod;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FederatedFSSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Partition data across clients (contiguous blocks; the last client takes the remainder)
        int baseSize = n / _nClients;
        var clientScores = new Tensor<T>[_nClients];
        var clientSizes = new int[_nClients];
        for (int c = 0; c < _nClients; c++)
        {
            int start = c * baseSize;
            int end = (c == _nClients - 1) ? n : (c + 1) * baseSize;
            clientSizes[c] = end - start;

            // Compute local feature scores using |correlation| (0 when sqrt(varX * varY) is not above 1e-10)
            var localX = engine.TensorSlice(X, new[] { start, 0 }, new[] { end - start, p });
            var localY = engine.TensorSlice(y, new[] { start, 0 }, new[] { end - start, 1 });
            clientScores[c] = engine.TensorAbs(engine.Reshape(FeatureSelectionTensorOps.ColumnCorrelations(localX, localY), new[] { 1, p }));
        }

        // Aggregate scores across clients ([clients, p])
        var scores = engine.TensorConcatenate(clientScores, 0);
        Tensor<T> aggregated;
        switch (_aggregationMethod)
        {
            case FederatedScoreAggregation.WeightedAverage:
                // Weight by client data size
                var sizes = new Tensor<int>(clientSizes, new[] { 1, _nClients }).Cast<T>();
                aggregated = engine.TensorDivideScalar(
                    engine.Reshape(engine.TensorMatMul(sizes, scores), new[] { p }), FeatureSelectionTensorOps.Count<T>(n));
                break;
            case FederatedScoreAggregation.Median:
                // Robust aggregation using the median
                aggregated = FeatureSelectionTensorOps.ColumnQuantile(scores, 0.5);
                break;
            case FederatedScoreAggregation.TrimmedMean:
                // Remove extreme values: drop the trimCount lowest and highest client scores
                var sorted = engine.TensorSort(scores, 0, false).Item1;
                int trimCount = Math.Max(1, _nClients / 5);
                int kept = _nClients - 2 * trimCount;
                aggregated = kept > 0
                    ? engine.ReduceMean(engine.TensorSlice(sorted, new[] { trimCount, 0 }, new[] { kept, p }), new[] { 0 }, false)
                    : engine.Reshape(engine.TensorSlice(sorted, new[] { _nClients / 2, 0 }, new[] { 1, p }), new[] { p });
                break;
            default:
                // Simple average
                aggregated = engine.ReduceMean(scores, new[] { 0 }, false);
                break;
        }

        _aggregatedScores = aggregated.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(aggregated, numToSelect);

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
            throw new InvalidOperationException("FederatedFSSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FederatedFSSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FederatedFSSelector has not been fitted.");

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
