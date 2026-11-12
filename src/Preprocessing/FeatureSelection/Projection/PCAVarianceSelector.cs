using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// PCA Variance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to principal components,
/// measuring how much variance each feature contributes to the data.
/// </para>
/// <para><b>For Beginners:</b> PCA finds the main directions of variation in data.
/// This selector identifies which original features contribute most to those
/// main directions, keeping features that explain the most variance.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PCAVarianceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;

    private Vector<T>? _varianceContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NComponents => _nComponents;
    public Vector<T>? VarianceContributions => _varianceContributions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PCAVarianceSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var engine = Engine;

        // Top principal axes of the sample covariance (exact symmetric eigendecomposition)
        int numComponents = Math.Min(_nComponents, p);
        var covariance = FeatureSelectionTensorOps.SampleCovariance(Tensor<T>.FromMatrix(data));
        var loadings = FeatureSelectionTensorOps.TopEigenpairs(covariance, numComponents, out var eigenvalues); // [p, c]

        // Contribution of each feature: squared loadings weighted by the share of the selected components'
        // variance each component explains (equal weights when that variance is not positive)
        var scores = new Tensor<T>(new[] { p });
        if (numComponents > 0)
        {
            T totalVariance = engine.TensorSum(eigenvalues);
            var weights = NumOps.GreaterThan(totalVariance, NumOps.Zero)
                ? engine.TensorDivideScalar(eigenvalues, totalVariance)
                : FeatureSelectionTensorOps.Filled(new[] { numComponents }, NumOps.FromDouble(1.0 / numComponents));
            scores = engine.Reshape(
                engine.TensorMatMul(engine.TensorSquare(loadings), engine.Reshape(weights, new[] { numComponents, 1 })), new[] { p });
        }

        _varianceContributions = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("PCAVarianceSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PCAVarianceSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PCAVarianceSelector has not been fitted.");

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
