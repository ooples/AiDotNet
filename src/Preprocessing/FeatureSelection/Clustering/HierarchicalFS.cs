using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Hierarchical Clustering-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses hierarchical (agglomerative) clustering to build a tree of feature
/// relationships, then cuts the tree to create groups and selects
/// representative features from each group.
/// </para>
/// <para><b>For Beginners:</b> This method builds a family tree of features,
/// starting with each feature as its own group, then gradually merging the
/// most similar ones. When we cut this tree at a certain height, we get
/// groups of related features. We pick the best feature from each group.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HierarchicalFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly string _linkage;

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

    public HierarchicalFS(
        int nFeaturesToSelect = 10,
        string linkage = "average",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        var validLinkages = new[] { "single", "complete", "average" };
        var normalizedLinkage = linkage.ToLowerInvariant();
        if (!validLinkages.Contains(normalizedLinkage))
            throw new ArgumentException(
                $"Unknown linkage method '{linkage}'. Valid options are: {string.Join(", ", validLinkages)}.",
                nameof(linkage));

        _nFeaturesToSelect = nFeaturesToSelect;
        _linkage = normalizedLinkage;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 2)
            throw new ArgumentException("HierarchicalFS requires at least 2 rows to compute variance/correlation.", nameof(data));

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Compute feature distance matrix based on correlation
        var distances = ComputeFeatureDistances(X, n, p);

        // Perform hierarchical clustering
        var clusterAssignments = AgglomerativeClustering(distances, p, _nFeaturesToSelect);

        // Compute feature scores (sample variance)
        var scores = engine.TensorDivideScalar(
            FeatureSelectionTensorOps.ColumnSumSquaredDeviations(X), NumOps.FromDouble(n - 1));
        _featureScores = scores.ToVector();

        // Select best feature from each cluster
        int numClusters = clusterAssignments.Max() + 1;
        _selectedIndices = FeatureSelectionTensorOps.BestPerGroup(scores, clusterAssignments, numClusters)
            .OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Tensor<T> ComputeFeatureDistances(Tensor<T> X, int n, int p)
    {
        var engine = Engine;

        // Compute means and stds (n - 1 denominator, plus 1e-10)
        var centered = engine.TensorSubtract(X, engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { n, p }));
        T dof = NumOps.FromDouble(n - 1);
        var stds = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, false), dof)),
            PreprocessingTensorOps.Tolerance<T>(1e-10));

        // Compute distance = 1 - |correlation|, 0 on the diagonal
        var corr = engine.TensorDivide(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered),
            engine.TensorMultiplyScalar(engine.TensorOuter(stds, stds), dof));
        var distances = PreprocessingTensorOps.OneMinus(engine.TensorAbs(corr));
        return engine.TensorWhere(engine.TensorEye<T>(p), new Tensor<T>(new[] { p, p }), distances);
    }

    private int[] AgglomerativeClustering(Tensor<T> distances, int p, int nClusters)
    {
        var engine = Engine;
        var assignments = Enumerable.Range(0, p).ToArray();
        var clusterSizes = new int[p];
        for (int j = 0; j < p; j++)
            clusterSizes[j] = 1;

        var clusterDistances = distances;
        int numActiveClusters = p;
        var shape = new[] { p, p };
        T maxValue = NumOps.MaxValue;
        var unreachable = PreprocessingTensorOps.Filled(shape, maxValue);
        var upper = engine.TensorTriu(PreprocessingTensorOps.Filled(shape, NumOps.One), 1);
        var positions = PreprocessingTensorOps.Range<T>(p);
        bool average = _linkage != "single" && _linkage != "complete";

        while (numActiveClusters > nClusters)
        {
            // Find closest pair of active clusters c1 < c2 (first pair in row-major order among ties)
            var active = new Tensor<int>(Array.ConvertAll(clusterSizes, s => s > 0 ? 1 : 0), new[] { p }).Cast<T>();
            var pairMask = engine.TensorMultiply(engine.TensorOuter(active, active), upper);
            var candidates = engine.TensorWhere(pairMask, clusterDistances, unreachable);
            if (!PreprocessingTensorOps.IsBelow(engine.TensorMinValue(candidates), maxValue))
                break;

            int flat = FeatureSelectionTensorOps.FirstArgMinPerRow(engine.Reshape(candidates, new[] { 1, p * p }))[0];
            int merge1 = flat / p;
            int merge2 = flat % p;

            // Merge clusters
            for (int j = 0; j < p; j++)
            {
                if (assignments[j] == merge2)
                    assignments[j] = merge1;
            }

            int newSize = clusterSizes[merge1] + clusterSizes[merge2];

            // Update distances using linkage method, for every other active cluster at once
            var row1 = PreprocessingTensorOps.RowVector(clusterDistances, merge1);
            var row2 = PreprocessingTensorOps.RowVector(clusterDistances, merge2);
            Tensor<T> newDist;
            if (average)
            {
                newDist = engine.TensorDivideScalar(
                    engine.TensorAdd(
                        engine.TensorMultiplyScalar(row1, PreprocessingTensorOps.CountAsScalar<T>(clusterSizes[merge1])),
                        engine.TensorMultiplyScalar(row2, PreprocessingTensorOps.CountAsScalar<T>(clusterSizes[merge2]))),
                    PreprocessingTensorOps.CountAsScalar<T>(newSize));
            }
            else
            {
                newDist = _linkage == "single" ? engine.TensorMin(row1, row2) : engine.TensorMax(row1, row2);
            }

            var isMerge1 = engine.TensorEquals(positions, PreprocessingTensorOps.CountAsScalar<T>(merge1));
            var isMerge2 = engine.TensorEquals(positions, PreprocessingTensorOps.CountAsScalar<T>(merge2));
            var others = engine.TensorMultiply(active, PreprocessingTensorOps.OneMinus(engine.TensorAdd(isMerge1, isMerge2)));
            clusterDistances = engine.TensorWhere(engine.TensorOuter(isMerge1, others),
                engine.TensorBroadcastTo(engine.Reshape(newDist, new[] { 1, p }), shape), clusterDistances);
            clusterDistances = engine.TensorWhere(engine.TensorOuter(others, isMerge1),
                engine.TensorBroadcastTo(engine.Reshape(newDist, new[] { p, 1 }), shape), clusterDistances);

            clusterSizes[merge1] = newSize;
            clusterSizes[merge2] = 0;
            numActiveClusters--;
        }

        // Renumber clusters from 0 to n-1
        var uniqueClusters = assignments.Distinct().OrderBy(c => c).ToList();
        var renumbering = new Dictionary<int, int>();
        for (int i = 0; i < uniqueClusters.Count; i++)
            renumbering[uniqueClusters[i]] = i;

        return assignments.Select(c => renumbering[c]).ToArray();
    }
    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HierarchicalFS has not been fitted.");

        if (data.Columns != _nInputFeatures)
            throw new ArgumentException($"Expected {_nInputFeatures} columns but got {data.Columns}.", nameof(data));

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HierarchicalFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HierarchicalFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HierarchicalFS has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (inputFeatureNames.Length < _nInputFeatures)
            throw new ArgumentException(
                $"Expected at least {_nInputFeatures} feature names, but got {inputFeatureNames.Length}.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
