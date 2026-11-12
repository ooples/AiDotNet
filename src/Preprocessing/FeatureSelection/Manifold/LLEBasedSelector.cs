using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Locally Linear Embedding (LLE) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Locally Linear Embedding to identify features that best preserve
/// the local neighborhood structure of the data manifold.
/// </para>
/// <para><b>For Beginners:</b> LLE assumes data lies on a curved surface
/// (manifold) in high-dimensional space. It finds how to reconstruct each
/// point from its neighbors. Features that are most important for these
/// local reconstructions are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LLEBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _regularization;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;

    /// <summary>
    /// Gets the regularization of the local Gram matrices, relative to their trace (scikit-learn's <c>reg</c>).
    /// </summary>
    public double Regularization => _regularization;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LLEBasedSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int[]? columnIndices = null,
        double regularization = 1e-3)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (regularization < 0)
            throw new ArgumentException("Regularization must be non-negative.", nameof(regularization));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _regularization = regularization;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        int k = Math.Min(_nNeighbors, n - 1);

        // Reconstruction weights from the k nearest neighbours (scikit-learn's barycenter weights)
        var weights = new Tensor<T>(new[] { n, n });
        if (k > 0)
        {
            var neighbors = FeatureSelectionTensorOps.KNearestMask(FeatureSelectionTensorOps.PairwiseSquaredDistances(x), k);
            weights = FeatureSelectionTensorOps.BarycenterWeights(
                x, FeatureSelectionTensorOps.NeighborIndices(neighbors, k), k, _regularization);
        }

        // Feature score: mean over the points of 1 / (squared reconstruction error + 1e-10)
        var error = engine.TensorSquare(engine.TensorSubtract(x, engine.TensorMatMul(weights, x)));
        var inverse = engine.TensorDivide(
            PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.One), engine.TensorAddScalar(error, NumOps.FromDouble(1e-10)));
        var scores = engine.ReduceMean(inverse, new[] { 0 }, false);
        _featureScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LLEBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LLEBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LLEBasedSelector has not been fitted.");

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
