using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.SemiSupervised;

/// <summary>
/// Semi-Supervised Feature Selection using Laplacian regularization.
/// </summary>
/// <remarks>
/// <para>
/// Combines supervised information from labeled samples with the manifold
/// structure captured from all samples (labeled and unlabeled) using the
/// graph Laplacian.
/// </para>
/// <para><b>For Beginners:</b> When you have few labeled samples but many
/// unlabeled ones, this method uses both. It builds a neighborhood graph
/// from all data to understand the data's structure, while using labels
/// where available.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SemiSupervisedLaplacian<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _alpha;

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

    public SemiSupervisedLaplacian(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double alpha = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        if (alpha < 0 || alpha > 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _alpha = alpha;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SemiSupervisedLaplacian requires partial labels. Use Fit with labeled mask.");
    }

    public void Fit(Matrix<T> data, Vector<T> target, bool[] isLabeled)
    {
        if (data.Rows != target.Length || data.Rows != isLabeled.Length)
            throw new ArgumentException("Data rows, target length, and label mask must match.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nLabeled = isLabeled.Count(l => l);

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Build k-NN graph and its degrees
        var w = BuildKNNGraph(x, n);
        var degrees = engine.ReduceSum(w, new[] { 1 }, false);

        // Unsupervised: Laplacian score f'Lf / f'Df of the degree-centred feature (0 when f'Df < 1e-10)
        var degreeColumn = engine.Reshape(degrees, new[] { n, 1 });
        T totalDegree = engine.TensorSum(degrees);
        var fBar = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorMultiply(x, degreeColumn), new[] { 0 }, true),
            NumOps.Add(totalDegree, NumOps.FromDouble(1e-10)));
        var centered = engine.TensorSubtract(x, fBar);
        var fDf = engine.ReduceSum(engine.TensorMultiply(engine.TensorMultiply(centered, centered), degreeColumn), new[] { 0 }, false);
        var fWf = engine.ReduceSum(engine.TensorMultiply(centered, engine.TensorMatMul(w, centered)), new[] { 0 }, false);
        var fLf = engine.TensorSubtract(fDf, fWf);
        var laplacianScores = SelectorTensorOps.ZeroUnless(
            PreprocessingTensorOps.AtLeast(fDf, NumOps.FromDouble(1e-10)), engine.TensorDivide(fLf, fDf));

        // Supervised: |correlation| with the labeled targets
        var supervisedScores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (nLabeled > 0)
        {
            var labeled = Enumerable.Range(0, n).Where(i => isLabeled[i]).ToArray();
            var rows = new Tensor<int>(labeled, new[] { labeled.Length });
            var labeledTargets = engine.Reshape(
                engine.TensorIndexSelect(Tensor<T>.FromVector(target), rows, 0), new[] { labeled.Length, 1 });
            supervisedScores = engine.TensorAbs(SelectorTensorOps.PearsonWithColumn(
                engine.TensorIndexSelect(x, rows, 0), labeledTargets));
        }

        // Combine scores
        var scores = engine.TensorAdd(
            engine.TensorMultiplyScalar(supervisedScores, NumOps.FromDouble(_alpha)),
            engine.TensorMultiplyScalar(PreprocessingTensorOps.OneMinus(laplacianScores), NumOps.FromDouble(1 - _alpha)));
        _featureScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    /// <summary>
    /// Heat-kernel k-NN graph: each sample links to its k nearest others (ties by index), weighted
    /// exp(-d^2 / (2 sigma^2)) with sigma its k-th neighbour distance + 1e-10. A pair linked from both ends keeps
    /// the weight from the higher-indexed sample, as the symmetric writes in row order leave it.
    /// </summary>
    private Tensor<T> BuildKNNGraph(Tensor<T> x, int n)
    {
        var engine = Engine;
        var shape = new[] { n, n };
        int k = Math.Min(_nNeighbors, n - 1);
        if (k <= 0)
            return new Tensor<T>(shape);

        // Pairwise Euclidean distances, self excluded
        var squaredNorms = engine.ReduceSum(engine.TensorMultiply(x, x), new[] { 1 }, false);
        var squared = engine.TensorSubtract(
            engine.TensorAdd(
                engine.TensorBroadcastTo(engine.Reshape(squaredNorms, new[] { n, 1 }), shape),
                engine.TensorBroadcastTo(engine.Reshape(squaredNorms, new[] { 1, n }), shape)),
            engine.TensorMultiplyScalar(engine.TensorMatMul(x, engine.TensorTranspose(x)), NumOps.FromDouble(2)));
        var distances = engine.TensorSqrt(engine.TensorMax(squared, NumOps.Zero));
        var eye = engine.TensorEye<T>(n);
        var others = engine.TensorWhere(eye, PreprocessingTensorOps.Filled(shape, NumOps.FromDouble(double.PositiveInfinity)), distances);

        // k nearest per row: everything below the k-th distance plus the lowest-indexed ties at it
        var kth = engine.TensorSliceAxis(engine.TensorSort(others, 1, false).Values, 1, k - 1);
        var kthColumn = engine.TensorBroadcastTo(engine.Reshape(kth, new[] { n, 1 }), shape);
        var below = engine.TensorLessThan(others, kthColumn);
        var ties = engine.TensorEquals(others, kthColumn);
        var room = engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { n, 1 }, NumOps.FromDouble(k)),
            engine.ReduceSum(below, new[] { 1 }, true));
        var tieRank = engine.TensorCumSum(ties, 1);
        var keptTies = engine.TensorMultiply(ties,
            engine.TensorLessEqual(tieRank, engine.TensorBroadcastTo(room, shape)));
        var adjacency = engine.TensorAdd(below, keptTies);

        // Row-wise heat kernel with sigma_i = k-th neighbour distance + 1e-10
        var sigma = engine.TensorAddScalar(kthColumn, NumOps.FromDouble(1e-10));
        var rowWeights = engine.TensorMultiply(adjacency, engine.TensorExp(engine.TensorNegate(engine.TensorDivide(
            engine.TensorMultiply(distances, distances),
            engine.TensorMultiplyScalar(engine.TensorMultiply(sigma, sigma), NumOps.FromDouble(2))))));

        // Symmetrize: entry (i, j) takes row max(i, j)'s weight when that row links the pair, else row min(i, j)'s
        var rowWeightsT = engine.TensorTranspose(rowWeights);
        var adjacencyT = engine.TensorTranspose(adjacency);
        var upper = SelectorTensorOps.StrictUpperMask<T>(n);
        return engine.TensorWhere(upper,
            engine.TensorWhere(adjacencyT, rowWeightsT, rowWeights),
            engine.TensorWhere(adjacency, rowWeights, rowWeightsT));
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target, bool[] isLabeled)
    {
        Fit(data, target, isLabeled);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SemiSupervisedLaplacian has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SemiSupervisedLaplacian does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SemiSupervisedLaplacian has not been fitted.");

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
