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
/// <para>
/// The score is phi3 of Zhao and Liu (2007), "Spectral Feature Selection for Supervised and Unsupervised
/// Learning" (ICML): for the normalised Laplacian <c>L = I - D^-1/2 W D^-1/2</c> of a k-nearest-neighbour Gaussian
/// graph, <c>sum_{j=2..k} (2 - lambda_j) (f^ . xi_j)^2</c> over its smallest eigenpairs (eigenvalues ascending, the
/// trivial first one skipped) with <c>f^ = D^1/2 f / ||D^1/2 f||</c>; <c>k</c> is <c>nEigenvectors</c>. Features
/// that vary smoothly over the graph score highest.
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

        // Zhao and Liu (2007), "Spectral Feature Selection for Supervised and Unsupervised Learning", score phi3:
        // sum over the eigenpairs j = 2..k of L in ascending eigenvalue order of (2 - lambda_j) (f^ . xi_j)^2, with
        // f^ = D^1/2 f / ||D^1/2 f|| (the first eigenpair, lambda = 0 and xi proportional to D^1/2 1, is skipped).
        // Larger is better. k is nEigenvectors, capped at n.
        int k = Math.Min(_nEigenvectors, n);
        Tensor<T> scores;
        if (k < 2 || p == 0)
        {
            scores = new Tensor<T>(new[] { p });
        }
        else
        {
            var (values, vectors) = LinalgDispatch<T>.Eigh(L);
            var index = engine.TensorArange<int>(1, k, 1);
            var xi = engine.TensorIndexSelect(vectors, index, 1);
            var weight = engine.TensorSubtract(
                FilterStatisticTensorOps.Filled(new[] { k - 1 }, NumOps.FromDouble(2)), engine.TensorIndexSelect(values, index, 0));

            // D^1/2 f, scaled by its largest magnitude before the norm (f^ does not change, and the squares stay
            // inside Half's range); a feature that is 0 on every connected row has no direction and scores 0
            var weighted = engine.TensorMultiply(x, FilterStatisticTensorOps.ColumnBroadcast(engine.TensorSqrt(D), p));
            var largest = PreprocessingTensorOps.ColumnMax(engine.TensorAbs(weighted));
            var scaled = FilterStatisticTensorOps.GuardedDivide(
                weighted, FilterStatisticTensorOps.RowBroadcast(largest, n), 0, NumOps.Zero);
            var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(scaled), new[] { 0 }, false));
            var unit = FilterStatisticTensorOps.GuardedDivide(scaled, FilterStatisticTensorOps.RowBroadcast(norm, n), 0, NumOps.Zero);
            var alpha = engine.TensorMatMul(engine.TensorTranspose(unit), xi);
            scores = engine.ReduceSum(
                engine.TensorMultiply(engine.TensorSquare(alpha), FilterStatisticTensorOps.RowBroadcast(weight, p)),
                new[] { 1 }, false);
        }

        _specScores = scores.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
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
