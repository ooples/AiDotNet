using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Correlation-based Feature Removal.
/// </summary>
/// <remarks>
/// <para>
/// Removes highly correlated features to reduce multicollinearity, keeping
/// one feature from each group of highly correlated features.
/// </para>
/// <para><b>For Beginners:</b> When two features are highly correlated, they
/// provide similar information. This method finds pairs of highly correlated
/// features and removes one from each pair, keeping your dataset compact
/// without losing much information.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CorrelationRemovalSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _threshold;

    private Matrix<T>? _correlationMatrix;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public Matrix<T>? CorrelationMatrix => _correlationMatrix;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CorrelationRemovalSelector(
        double threshold = 0.9,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;

        // Correlation matrix (sample covariance over sample std + 1e-10, unit diagonal)
        var correlation = FeatureSelectionTensorOps.SampleCorrelationMatrix(Tensor<T>.FromMatrix(data), 1e-10);
        _correlationMatrix = correlation.ToMatrix();

        // Walking the features in order, each kept feature removes every later one correlated above the threshold
        var redundant = engine.TensorGreaterThan(engine.TensorAbs(correlation), NumOps.FromDouble(_threshold));
        _selectedIndices = FeatureSelectionTensorOps.KeepFirstOfRedundantPairs(redundant);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationRemovalSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CorrelationRemovalSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationRemovalSelector has not been fitted.");

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
