using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// SVD-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Singular Value Decomposition to identify features that contribute
/// most to the dominant singular vectors.
/// </para>
/// <para><b>For Beginners:</b> SVD breaks down your data into components ordered
/// by importance. Features that have high values in the most important components
/// are the features that capture the most information about the data's structure.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SVDBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SVDBasedSelector(
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
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Top right singular vectors V of the centred data, from the eigendecomposition of Xc'Xc
        // (whose eigenvalues are the squared singular values)
        int k = Math.Min(_nComponents, Math.Min(n, p));
        var centered = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var gram = engine.TensorMatMul(engine.TensorTranspose(centered), centered);
        var v = FeatureSelectionTensorOps.TopEigenpairs(gram, k, out var squaredSingularValues); // [p, k]

        // Feature scores = squared loadings weighted by each component's share of the variance
        var scores = new Tensor<T>(new[] { p });
        if (k > 0)
        {
            var squared = engine.TensorMax(squaredSingularValues, NumOps.Zero);
            T totalVar = NumOps.Add(engine.TensorSum(squared), NumOps.FromDouble(1e-10));
            var weights = engine.TensorDivideScalar(squared, totalVar);
            scores = engine.Reshape(
                engine.TensorMatMul(engine.TensorSquare(v), engine.Reshape(weights, new[] { k, 1 })), new[] { p });
        }

        _featureScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SVDBasedSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SVDBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SVDBasedSelector has not been fitted.");

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
