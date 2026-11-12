using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Random Projection-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses random projections to identify features that contribute most to
/// preserving distances in the projected space (Johnson-Lindenstrauss lemma).
/// </para>
/// <para><b>For Beginners:</b> Random projections preserve important structure
/// in data when projecting to lower dimensions. By looking at which original
/// features contribute most to these projections, we can identify the most
/// important features without expensive computations.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RandomProjectionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nProjections;
    private readonly int _projectionDim;

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

    public RandomProjectionSelector(
        int nFeaturesToSelect = 10,
        int nProjections = 50,
        int projectionDim = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nProjections = nProjections;
        _projectionDim = projectionDim;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        int k = Math.Min(_projectionDim, p);
        T oneSixth = NumOps.FromDouble(1.0 / 6);
        T oneThird = NumOps.FromDouble(2.0 / 6);
        T sqrtP = NumOps.FromDouble(Math.Sqrt(p));
        T sampleDivisor = NumOps.FromDouble(n - 1);
        var scores = new Tensor<T>(new[] { p });

        for (int proj = 0; proj < _nProjections; proj++)
        {
            // Sparse (Achlioptas) random projection matrix: +sqrt(p) and -sqrt(p) with probability 1/6 each,
            // 0 with probability 4/6
            var u = engine.TensorRandomUniformRange(new[] { p, k }, NumOps.Zero, NumOps.One, rand.Next());
            var positive = engine.TensorLessThan(u, oneSixth);
            var nonZero = engine.TensorLessThan(u, oneThird);
            var negative = engine.TensorSubtract(nonZero, positive);
            var r = engine.TensorMultiplyScalar(engine.TensorSubtract(positive, negative), sqrtP);

            // Sample variance of each projected dimension
            var projected = engine.TensorMatMul(x, r);
            var deviations = engine.TensorSubtract(projected, engine.ReduceMean(projected, new[] { 0 }, true));
            var variance = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(deviations), new[] { 0 }, false), sampleDivisor);

            // Weight features by their contribution to high-variance projections
            var absR = engine.TensorMultiplyScalar(nonZero, sqrtP);
            scores = engine.TensorAdd(scores, engine.Reshape(engine.TensorMatMul(absR, engine.Reshape(variance, new[] { k, 1 })), new[] { p }));
        }

        // Normalize
        if (p > 0)
        {
            T maxScore = engine.TensorMaxValue(scores);
            if (NumOps.GreaterThan(maxScore, NumOps.Zero))
                scores = engine.TensorDivideScalar(scores, maxScore);
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
            throw new InvalidOperationException("RandomProjectionSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RandomProjectionSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomProjectionSelector has not been fitted.");

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
