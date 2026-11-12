using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Redundancy;

/// <summary>
/// Correlation Cluster based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features by clustering correlated features together and selecting
/// the best representative from each cluster.
/// </para>
/// <para><b>For Beginners:</b> Features that are highly correlated form natural
/// groups. Instead of keeping all similar features, this selector picks one
/// representative from each group, reducing redundancy while keeping diversity.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CorrelationClusterSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _correlationThreshold;

    private Vector<T>? _clusterRepresentativeScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CorrelationThreshold => _correlationThreshold;
    public Vector<T>? ClusterRepresentativeScores => _clusterRepresentativeScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CorrelationClusterSelector(
        int nFeaturesToSelect = 10,
        double correlationThreshold = 0.8,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _correlationThreshold = correlationThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CorrelationClusterSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var x = Tensor<T>.FromMatrix(data);

        // Relevance |corr(feature, target)| and the feature-feature |correlation| matrix
        var relevance = engine.TensorAbs(engine.Reshape(
            FeatureSelectionTensorOps.ColumnCorrelations(x, engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })), new[] { p }));
        var correlation = engine.TensorAbs(FeatureSelectionTensorOps.ColumnCorrelations(x, x));
        T threshold = NumOps.FromDouble(_correlationThreshold);

        // Agglomerative clustering in order of relevance (most relevant first): each feature not yet in a
        // cluster leads a new one and absorbs every unassigned feature correlated with it at the threshold or
        // more. A leader is the most relevant member of its cluster, so the leaders, in that order, are the
        // cluster representatives.
        var order = FeatureSelectionTensorOps.StableOrder(relevance, descending: true);
        var assigned = new Tensor<T>(new[] { p });
        var representatives = new List<int>();
        int position = 0;
        while (representatives.Count < _nFeaturesToSelect)
        {
            var taken = assigned.Cast<int>().AsSpan().ToArray();
            while (position < order.Length && taken[order[position]] != 0)
                position++;
            if (position >= order.Length)
                break;

            int leader = order[position];
            representatives.Add(leader);
            var absorbed = FeatureSelectionTensorOps.OneMinus(engine.TensorLessThan(
                engine.Reshape(engine.TensorSlice(correlation, new[] { leader, 0 }, new[] { 1, p }), new[] { p }), threshold));
            assigned = engine.TensorMax(engine.TensorMax(assigned, absorbed), FeatureSelectionTensorOps.OneHot<T>(leader, p));
        }

        var isRepresentative = new Tensor<T>(new[] { p });
        foreach (int leader in representatives)
            isRepresentative = engine.TensorAdd(isRepresentative, FeatureSelectionTensorOps.OneHot<T>(leader, p));
        _clusterRepresentativeScores = engine.TensorMultiply(isRepresentative, relevance).ToVector();
        _selectedIndices = representatives.OrderBy(index => index).ToArray();

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
            throw new InvalidOperationException("CorrelationClusterSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CorrelationClusterSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationClusterSelector has not been fitted.");

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
