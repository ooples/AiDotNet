using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Density;

/// <summary>
/// Local Outlier Factor (LOF) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Local Outlier Factor concept to select features that best preserve
/// local density structure, helping maintain meaningful neighborhoods.
/// </para>
/// <para><b>For Beginners:</b> LOF measures how "outlying" each point is compared
/// to its neighbors. This selector chooses features that help distinguish normal
/// points from outliers, preserving the local density patterns in your data.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LOFBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private Vector<T>? _lofContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public Vector<T>? LOFContributions => _lofContributions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LOFBasedSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        int k = Math.Min(_nNeighbors, n - 1);

        // Compute LOF scores using all features
        var lofVarianceAll = ComputeVariance(ComputeLOF(X, n, k));

        // For each feature, compute LOF without it and measure change
        var contributions = new Tensor<T>[p];
        for (int excludeJ = 0; excludeJ < p; excludeJ++)
        {
            int excluded = excludeJ;
            var kept = Enumerable.Range(0, p).Where(j => j != excluded).ToArray();
            var XReduced = engine.TensorIndexSelect(X, new Tensor<int>(kept, new[] { kept.Length }), 1);

            var lofVarianceReduced = ComputeVariance(ComputeLOF(XReduced, n, k));

            // Feature importance = how much LOF variance changes when removed
            contributions[excludeJ] = engine.TensorAbs(engine.TensorSubtract(lofVarianceAll, lofVarianceReduced));
        }

        var scores = p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(contributions, 0);
        _lofContributions = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeLOF(Tensor<T> X, int n, int k)
    {
        var engine = Engine;

        // Compute distance matrix
        var distances = X.Shape[1] == 0
            ? new Tensor<T>(new[] { n, n })
            : engine.TensorSqrt(FeatureSelectionTensorOps.SquaredDistances(X, X));

        // Find k-nearest neighbors (closest first, lower index among ties) and k-distance
        var maxed = engine.TensorWhere(engine.TensorEye<T>(n),
            PreprocessingTensorOps.Filled(new[] { n, n }, NumOps.MaxValue), distances);
        var neighborFlags = new int[n * n];
        var kthNeighbor = new int[n];
        for (int i = 0; i < n; i++)
        {
            var order = PreprocessingTensorOps.StableOrder(PreprocessingTensorOps.RowVector(maxed, i), descending: false);
            for (int r = 0; r < k; r++)
                neighborFlags[i * n + order[r]] = 1;
            kthNeighbor[i] = i * n + order[k - 1];
        }

        var neighbors = new Tensor<int>(neighborFlags, new[] { n, n }).Cast<T>();
        var kDistances = engine.TensorIndexSelect(
            engine.Reshape(distances, new[] { n * n }), new Tensor<int>(kthNeighbor, new[] { n }), 0);

        // Compute reachability distances: max(k-distance of j, d(i, j))
        var reachDist = engine.TensorMax(
            engine.TensorBroadcastTo(engine.Reshape(kDistances, new[] { 1, n }), new[] { n, n }), distances);

        // Compute local reachability density (LRD)
        T kScalar = NumOps.FromDouble(k);
        T epsilon = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var reachSums = engine.ReduceSum(engine.TensorMultiply(neighbors, reachDist), new[] { 1 }, false);
        var lrd = engine.TensorDivide(
            PreprocessingTensorOps.Filled(new[] { n }, kScalar), engine.TensorAddScalar(reachSums, epsilon));

        // Compute LOF
        var neighborLrd = engine.Reshape(engine.TensorMatMul(neighbors, engine.Reshape(lrd, new[] { n, 1 })), new[] { n });
        return engine.TensorDivideScalar(engine.TensorDivide(neighborLrd, engine.TensorAddScalar(lrd, epsilon)), kScalar);
    }

    private Tensor<T> ComputeVariance(Tensor<T> values)
    {
        var engine = Engine;
        var deviations = engine.TensorSubtract(values, engine.TensorBroadcastTo(
            engine.ReduceMean(values, new[] { 0 }, true), values.Shape.ToArray()));
        return engine.ReduceMean(engine.TensorSquare(deviations), new[] { 0 }, true);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LOFBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LOFBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LOFBasedSelector has not been fitted.");

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
