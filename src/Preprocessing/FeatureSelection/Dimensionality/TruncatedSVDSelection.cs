using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Dimensionality;

/// <summary>
/// Truncated SVD Feature Selection using singular value decomposition.
/// </summary>
/// <remarks>
/// <para>
/// Uses Singular Value Decomposition to identify which original features
/// contribute most to the principal directions of variation. Unlike PCA,
/// this works directly on the data matrix without centering.
/// </para>
/// <para><b>For Beginners:</b> SVD breaks down your data into components
/// that explain different directions of variation. By looking at which
/// original features contribute to the most important directions, we can
/// select features that capture the most information about your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TruncatedSVDSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
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

    public TruncatedSVDSelection(
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

        // Compute X^T X for right singular vectors
        var xTx = engine.TensorMatMul(engine.TensorTranspose(X), X);

        // Power iteration for top eigenvectors of X^T X (negative eigenvalues floored at 0 as a numerical guard)
        int nComp = Math.Min(_nComponents, p);
        var loadings = FeatureSelectionTensorOps.PowerIterationComponents(xTx, nComp, 100, true, out var eigenvalues);
        var singularValues = engine.TensorSqrt(eigenvalues);

        // Compute feature importance as weighted sum of squared loadings
        var squared = engine.TensorSquare(singularValues);
        var weights = engine.TensorDivideScalar(squared, NumOps.Add(engine.TensorSum(squared), NumOps.FromDouble(1e-10)));
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
            throw new InvalidOperationException("TruncatedSVDSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TruncatedSVDSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TruncatedSVDSelection has not been fitted.");

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
