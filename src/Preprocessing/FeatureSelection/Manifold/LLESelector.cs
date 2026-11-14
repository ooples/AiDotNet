using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Locally Linear Embedding (LLE) inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their importance in preserving local linear
/// relationships in the data manifold.
/// </para>
/// <para><b>For Beginners:</b> LLE assumes each point can be described as a
/// linear combination of its neighbors. This selector finds features important
/// for maintaining those local linear relationships.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LLESelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _regularization;

    private Vector<T>? _localLinearityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public Vector<T>? LocalLinearityScores => _localLinearityScores;

    /// <summary>
    /// Gets the regularization of the local Gram matrices, relative to their trace (scikit-learn's <c>reg</c>).
    /// </summary>
    public double Regularization => _regularization;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LLESelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
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

        // Points need at least 2 neighbours to be reconstructed; otherwise every score is 0
        var scores = new Tensor<T>(new[] { p });
        if (k >= 2)
        {
            // LLE reconstructs each point as a weighted (barycenter) combination of its k nearest neighbours
            var neighbors = FeatureSelectionTensorOps.KNearestMask(FeatureSelectionTensorOps.PairwiseDistances(x), k);
            var weights = FeatureSelectionTensorOps.BarycenterWeights(
                x, FeatureSelectionTensorOps.NeighborIndices(neighbors, k), k, _regularization);

            // Features with low reconstruction error are more "locally linear": mean of 1 / (1 + |error|)
            var error = engine.TensorAbs(engine.TensorSubtract(x, engine.TensorMatMul(weights, x)));
            var closeness = engine.TensorDivide(
                PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.One), engine.TensorAddScalar(error, NumOps.One));
            scores = engine.ReduceMean(closeness, new[] { 0 }, false);
        }

        _localLinearityScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("LLESelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LLESelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LLESelector has not been fitted.");

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
