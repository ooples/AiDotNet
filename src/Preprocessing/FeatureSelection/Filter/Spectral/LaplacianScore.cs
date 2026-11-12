using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Laplacian Score for unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Laplacian Score evaluates features based on their ability to preserve locality
/// in the data. Features with low Laplacian Score preserve local structure well,
/// meaning similar samples have similar feature values.
/// </para>
/// <para><b>For Beginners:</b> The Laplacian Score measures how smoothly a feature
/// varies across similar data points. A good feature should have similar values
/// for nearby points (like neighbors having similar house prices). Features that
/// jump around randomly between neighbors score poorly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LaplacianScore<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _kernelWidth;

    private Vector<T>? _laplacianScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public Vector<T>? LaplacianScores => _laplacianScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LaplacianScore(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double kernelWidth = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _kernelWidth = kernelWidth;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // k-nearest-neighbour graph with heat-kernel weights exp(-||x_i - x_j||^2 / t^2), symmetric
        var weights = FeatureSelectionTensorOps.KnnGaussianAffinity(x, _nNeighbors, 1.0 / (_kernelWidth * _kernelWidth));
        var degree = engine.ReduceSum(weights, new[] { 1 }, false);

        // f~ = f - (f'D1 / 1'D1) 1, score = f~' L f~ / f~' D f~ with L = D - W (He, Cai and Niyogi 2005)
        var degreeRow = engine.Reshape(degree, new[] { 1, n });
        var mean = engine.TensorDivideScalar(
            engine.Reshape(engine.TensorMatMul(degreeRow, x), new[] { p }), engine.TensorSum(degree));
        var centered = engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(mean, n));
        var denominator = engine.Reshape(engine.TensorMatMul(degreeRow, engine.TensorSquare(centered)), new[] { p });
        var smoothness = engine.ReduceSum(
            engine.TensorMultiply(centered, engine.TensorMatMul(weights, centered)), new[] { 0 }, false);
        var numerator = engine.TensorSubtract(denominator, smoothness);
        var scores = FilterStatisticTensorOps.GuardedDivide(numerator, denominator, 1e-10, NumOps.MaxValue);
        _laplacianScores = scores.ToVector();

        // Lower is better
        _selectedIndices = FilterStatisticTensorOps.BottomKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScore has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LaplacianScore does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScore has not been fitted.");

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
