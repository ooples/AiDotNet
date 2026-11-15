using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Multi-Label Mutual Information (ML-MI) for multi-label feature selection.
/// </summary>
/// <remarks>
/// <para>
/// ML-MI extends mutual information to handle multi-label problems by computing
/// the mutual information between each feature and the entire label set. It can
/// consider label correlations and dependencies.
/// </para>
/// <para><b>For Beginners:</b> Mutual information measures how much knowing one
/// thing tells you about another. In multi-label problems, ML-MI measures how
/// much each feature tells you about the entire combination of labels an instance
/// might have.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MLMI<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _considerLabelCorrelations;

    private Vector<T>? _mutualInformation;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public bool ConsiderLabelCorrelations => _considerLabelCorrelations;
    public Vector<T>? MutualInformation => _mutualInformation;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MLMI(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        bool considerLabelCorrelations = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _considerLabelCorrelations = considerLabelCorrelations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MLMI requires label matrix. Use Fit(Matrix<T> data, Matrix<T> labels) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> labels)
    {
        if (data.Rows != labels.Rows)
            throw new ArgumentException("Number of samples must match between data and labels.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int numLabels = labels.Columns;

        var engine = Engine;

        // Convert labels to binary, and discretize the features into nBins equal-width bins
        // (numpy.histogram bins; a constant feature is one bin)
        var binaryLabels = engine.TensorGreaterThan(Tensor<T>.FromMatrix(labels), NumOps.FromDouble(0.5)).Cast<int>();
        var discretized = FeatureSelectionInformationOps.EqualWidthBinsExactRange(Tensor<T>.FromMatrix(data), _nBins);

        Tensor<T> mutualInformation;
        if (_considerLabelCorrelations)
        {
            // Compute MI (nats) with the label powerset: each distinct label combination is one class
            var combination = new Tensor<int>(new[] { n });
            int combinations = 1;
            for (int l = 0; l < numLabels; l++)
            {
                var label = engine.Reshape(engine.TensorSlice(binaryLabels, new[] { 0, l }, new[] { n, 1 }), new[] { n });
                var code = engine.TensorAdd(engine.TensorMultiplyScalar(combination, 2), label);
                var (values, inverse, _) = engine.TensorUniqueWithInfo(code, true, true, false);
                combination = inverse ?? throw new InvalidOperationException("The unique label combinations were not returned.");
                combinations = values.Length;
            }

            mutualInformation = FeatureSelectionInformationOps.MutualInformation(
                FeatureSelectionInformationOps.JointCounts<T>(discretized, _nBins, combination, combinations), n);
        }
        else
        {
            // Mean over the labels of the MI (nats) with each single label
            var total = new Tensor<T>(new[] { p });
            for (int l = 0; l < numLabels; l++)
            {
                var label = engine.TensorSlice(binaryLabels, new[] { 0, l }, new[] { n, 1 });
                total = engine.TensorAdd(total, FeatureSelectionInformationOps.MutualInformation(
                    FeatureSelectionInformationOps.JointCounts<T>(discretized, _nBins, label, 2), n));
            }

            mutualInformation = engine.TensorDivideScalar(total, FeatureSelectionTensorOps.Count<T>(numLabels));
        }

        _mutualInformation = mutualInformation.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(mutualInformation, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        // Convert single-label to multi-label format: one indicator column per distinct (rounded) label, ascending
        var classIds = FeatureSelectionInformationOps.ClassIndices(Engine.TensorRound(Tensor<T>.FromVector(target)), out int numLabels);
        var indicators = Engine.TensorOneHot<T>(classIds, numLabels);
        Fit(data, indicators.ToMatrix());
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> labels)
    {
        Fit(data, labels);
        return Transform(data);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MLMI has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MLMI does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MLMI has not been fitted.");

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
