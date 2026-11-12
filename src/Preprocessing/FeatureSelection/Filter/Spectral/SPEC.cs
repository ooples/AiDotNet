using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// SPEC (Spectral Feature Selection) for unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SPEC ranks features based on their consistency with the structure of the data
/// as captured by a similarity graph's spectral representation. It uses the
/// eigenvectors of the graph Laplacian to evaluate feature quality.
/// </para>
/// <para><b>For Beginners:</b> SPEC builds a graph where similar data points are
/// connected. It then looks at the graph's "shape" (via eigenvectors) and asks:
/// which features align well with this structure? Features that respect the natural
/// groupings in the data get high scores.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SPEC<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _sigma;
    private readonly int _nEigenvectors;

    private Vector<T>? _specScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SpecScores => _specScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SPEC(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double sigma = 1.0,
        int nEigenvectors = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _sigma = sigma;
        _nEigenvectors = nEigenvectors;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // k-nearest-neighbour graph with Gaussian weights exp(-d^2 / (2 sigma^2))
        var W = FeatureSelectionTensorOps.KnnGaussianAffinity(x, _nNeighbors, 1.0 / (2 * _sigma * _sigma));
        var D = engine.ReduceSum(W, new[] { 1 }, false);

        // normalised Laplacian L = I - D^-1/2 W D^-1/2 (rows without edges keep only the identity)
        var dInvSqrt = FilterStatisticTensorOps.GuardedDivide(
            FilterStatisticTensorOps.FilledLike(D, NumOps.One), engine.TensorSqrt(D), Math.Sqrt(1e-10), NumOps.Zero);
        var L = engine.TensorSubtract(engine.TensorEye<T>(n), engine.TensorMultiply(W, engine.TensorOuter(dInvSqrt, dInvSqrt)));

        var eigenvectors = ComputeTopEigenvectors(L, n, Math.Min(_nEigenvectors, n - 1));

        // standardised features (population std) projected on the eigenvectors: score = sum_k (f' v_k)^2
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var std = engine.TensorSqrt(FilterStatisticTensorOps.ColumnPopulationVariance(x));
        var scale = FilterStatisticTensorOps.GuardedDivide(FilterStatisticTensorOps.FilledLike(std, NumOps.One), std, 1e-10, NumOps.One);
        var standardized = engine.TensorMultiply(centered, FilterStatisticTensorOps.RowBroadcast(scale, n));
        var scores = eigenvectors.Count == 0
            ? new Tensor<T>(new[] { p })
            : engine.ReduceSum(engine.TensorSquare(engine.TensorMatMul(
                engine.TensorTranspose(standardized), engine.TensorConcatenate(eigenvectors.ToArray(), 1))), new[] { 1 }, false);
        _specScores = scores.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, _nFeaturesToSelect);

        IsFitted = true;
    }

    // Power iteration (50 steps from the uniform vector) with deflation against the vectors already found; each
    // eigenvector is an [n, 1] tensor.
    private List<Tensor<T>> ComputeTopEigenvectors(Tensor<T> L, int n, int k)
    {
        var engine = Engine;
        var eigenvectors = new List<Tensor<T>>();
        if (k <= 0)
            return eigenvectors;

        var start = FilterStatisticTensorOps.Filled(new[] { n, 1 }, NumOps.FromDouble(1.0 / Math.Sqrt(n)));
        var tolerance = NumOps.FromDouble(1e-10);
        T one = NumOps.One;
        for (int ev = 0; ev < k; ev++)
        {
            var v = start;
            for (int iter = 0; iter < 50; iter++)
            {
                var Lv = engine.TensorMatMul(L, v);
                foreach (var prev in eigenvectors)
                {
                    var dot = engine.TensorMatMul(engine.TensorTranspose(prev), Lv);
                    Lv = engine.TensorSubtract(Lv, engine.TensorMultiply(prev, engine.TensorBroadcastTo(dot, new[] { n, 1 })));
                }

                var norm = engine.TensorBroadcastTo(
                    engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(Lv), new[] { 0 }, true)), new[] { n, 1 });
                var moved = engine.TensorGreaterThan(norm, tolerance);
                var safeNorm = engine.TensorWhere(moved, norm, FilterStatisticTensorOps.FilledLike(norm, one));
                v = engine.TensorWhere(moved, engine.TensorDivide(Lv, safeNorm), v);
            }

            eigenvectors.Add(v);
        }

        return eigenvectors;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SPEC has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SPEC does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SPEC has not been fitted.");

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
