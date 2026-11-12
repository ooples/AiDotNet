using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Dimensionality;

/// <summary>
/// PCA-Based Feature Selection using loadings analysis.
/// </summary>
/// <remarks>
/// <para>
/// Uses Principal Component Analysis to identify which original features
/// contribute most to the principal components. Features with high loadings
/// on important components are selected.
/// </para>
/// <para><b>For Beginners:</b> PCA finds combinations of features that capture
/// the most variation in your data. By looking at which original features
/// contribute most to these combinations, we can identify which features
/// are most informative overall.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PCABasedSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NComponents => _nComponents;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PCABasedSelection(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nComponents < 1)
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Center the data
        var centered = engine.TensorSubtract(X, engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { n, p }));

        // Compute covariance matrix
        var cov = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), NumOps.FromDouble(n - 1));

        // Power iteration for top eigenvalues/eigenvectors
        int nComp = Math.Min(_nComponents, p);
        var loadings = FeatureSelectionTensorOps.PowerIterationComponents(cov, nComp, 100, false, out var eigenvalues);

        // Compute feature importance as weighted sum of squared loadings
        var weights = engine.TensorDivideScalar(eigenvalues,
            NumOps.Add(engine.TensorSum(eigenvalues), PreprocessingTensorOps.Tolerance<T>(1e-10)));
        var importances = FeatureSelectionTensorOps.WeightedSquaredLoadings(loadings, weights);
        _featureImportances = importances.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importances, numToSelect);

        IsFitted = true;
    }
    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PCABasedSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PCABasedSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PCABasedSelection has not been fitted.");

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
