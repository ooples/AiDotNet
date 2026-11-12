using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Multi-Label Mutual Information for multi-label classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Extends mutual information to handle multiple target labels simultaneously.
/// Aggregates MI scores across all labels to identify features relevant to
/// multiple prediction targets.
/// </para>
/// <para><b>For Beginners:</b> In multi-label classification, each sample can
/// have multiple labels at once. This method finds features that are informative
/// for predicting multiple labels, not just one.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MultiLabelMutualInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly AggregationMethod _aggregation;

    public enum AggregationMethod { Sum, Average, Max, Min }

    private Vector<T>? _miScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? MIScores => _miScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MultiLabelMutualInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        AggregationMethod aggregation = AggregationMethod.Sum,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _aggregation = aggregation;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MultiLabelMutualInformation requires target values. Use Fit(Matrix<T> data, Matrix<T> targets) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> targets)
    {
        if (data.Rows != targets.Rows)
            throw new ArgumentException("Target rows must match data rows.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nLabels = targets.Columns;
        if (n == 0 && p > 0)
            throw new InvalidOperationException("Sequence contains no elements");
        if (nLabels == 0 && p > 0 && _aggregation != AggregationMethod.Sum)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;

        // Discretize features into nBins equal-width bins (numpy.histogram bins; a constant feature is one bin)
        // and the labels into binary values (> 0.5)
        var xDiscrete = FeatureSelectionInformationOps.EqualWidthBinsExactRange(Tensor<T>.FromMatrix(data), _nBins);
        var yDiscrete = engine.TensorGreaterThan(Tensor<T>.FromMatrix(targets), NumOps.FromDouble(0.5)).Cast<int>();

        // Compute MI (nats) for each feature-label pair, as [p, nLabels]
        var miScores = new Tensor<T>(new[] { p });
        if (nLabels > 0 && p > 0)
        {
            var columns = new Tensor<T>[nLabels];
            for (int l = 0; l < nLabels; l++)
            {
                var label = engine.TensorSlice(yDiscrete, new[] { 0, l }, new[] { n, 1 });
                columns[l] = engine.Reshape(FeatureSelectionInformationOps.MutualInformation(
                    FeatureSelectionInformationOps.JointCounts<T>(xDiscrete, _nBins, label, 2), n), new[] { p, 1 });
            }

            // Aggregate across labels
            var miMatrix = engine.TensorConcatenate(columns, 1);
            miScores = _aggregation switch
            {
                AggregationMethod.Average => engine.ReduceMean(miMatrix, new[] { 1 }, false),
                AggregationMethod.Max => FeatureSelectionTensorOps.ColumnMax(engine.TensorTranspose(miMatrix)),
                AggregationMethod.Min => FeatureSelectionTensorOps.ColumnMin(engine.TensorTranspose(miMatrix)),
                _ => engine.ReduceSum(miMatrix, new[] { 1 }, false)
            };
        }

        _miScores = miScores.ToVector();

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(miScores, nToSelect);

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> targets)
    {
        Fit(data, targets);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelMutualInformation has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MultiLabelMutualInformation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelMutualInformation has not been fitted.");

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
