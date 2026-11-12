using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Laplacian Score Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Laplacian Score to select features that best preserve the local
/// structure of the data. Features with lower Laplacian scores are considered
/// more important.
/// </para>
/// <para><b>For Beginners:</b> The Laplacian score measures how much a feature
/// changes between nearby points. Features that are smooth (don't change much
/// between neighbors) have low Laplacian scores and are considered more
/// important for preserving local structure.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LaplacianScoreSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _sigma;

    private Vector<T>? _laplacianScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? LaplacianScores => _laplacianScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LaplacianScoreSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        double sigma = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _sigma = sigma;
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

        // k-NN heat-kernel similarity graph, symmetrized (i ~ j when either is among the other's neighbours)
        var squaredDistances = FeatureSelectionTensorOps.PairwiseSquaredDistances(x);
        var neighbors = FeatureSelectionTensorOps.KNearestMask(squaredDistances, k);
        var adjacency = engine.TensorMax(neighbors, engine.TensorTranspose(neighbors));
        var heat = engine.TensorExp(engine.TensorDivideScalar(
            engine.TensorNegate(squaredDistances), NumOps.FromDouble(2 * _sigma * _sigma)));
        var w = engine.TensorMultiply(adjacency, heat);

        // Degrees D = W 1
        var degree = engine.ReduceSum(w, new[] { 1 }, true); // [n, 1]
        T degreeSum = engine.TensorSum(degree);

        // f~ = f - (f'D1 / 1'D1) 1 for every feature column f
        var weighted = engine.TensorMultiply(x, degree);
        var alpha = engine.TensorDivideScalar(
            engine.ReduceSum(weighted, new[] { 0 }, true), NumOps.Add(degreeSum, NumOps.FromDouble(1e-10)));
        var centered = engine.TensorSubtract(x, alpha);

        // Laplacian score = f~'L f~ / f~'D f~ with L = D - W
        var denominator = engine.ReduceSum(engine.TensorMultiply(engine.TensorSquare(centered), degree), new[] { 0 }, false);
        var smoothness = engine.ReduceSum(engine.TensorMultiply(centered, engine.TensorMatMul(w, centered)), new[] { 0 }, false);
        var numerator = engine.TensorSubtract(denominator, smoothness);
        var defined = engine.TensorGreaterThan(denominator, NumOps.FromDouble(1e-10));
        var safeDenominator = engine.TensorWhere(defined, denominator, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
        var scores = engine.TensorWhere(
            defined, engine.TensorDivide(numerator, safeDenominator), FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.MaxValue));
        _laplacianScores = scores.ToVector();

        // Select features with LOWEST Laplacian scores
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.BottomKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScoreSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LaplacianScoreSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScoreSelector has not been fitted.");

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
