using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Non-negative Matrix Factorization (NMF) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses NMF to decompose the data matrix and selects features based on
/// their contribution to the learned basis vectors.
/// </para>
/// <para><b>For Beginners:</b> NMF breaks down your data into parts (like
/// breaking a photo into its component patterns). Features that contribute
/// most to these fundamental patterns are the most informative ones to keep.
/// NMF is especially good when features represent non-negative quantities.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NMFBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;
    private readonly int _nIterations;

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

    public NMFBasedSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int nIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
        _nIterations = nIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        // Ensure non-negative
        var x = engine.TensorMax(Tensor<T>.FromMatrix(data), NumOps.Zero);

        int k = Math.Min(_nComponents, Math.Min(n, p));
        var rand = RandomHelper.CreateSecureRandom();

        // Initialize W (n x k) and H (k x p) with random non-negative values
        T offset = NumOps.FromDouble(0.01);
        var w = engine.TensorAddScalar(PopulationSearchOps.Uniform<T>(rand, n, k), offset);
        var h = engine.TensorAddScalar(PopulationSearchOps.Uniform<T>(rand, k, p), offset);
        T eps = PopulationSearchOps.Tiny<T>(1e-10);

        // Multiplicative update rules (Lee and Seung 2001)
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Update H: H = H .* (W'X) ./ (W'WH + eps)
            var wt = engine.TensorTranspose(w);
            var wtx = engine.TensorMatMul(wt, x);
            var wtwh = engine.TensorMatMul(engine.TensorMatMul(wt, w), h);
            h = engine.TensorMultiply(h, engine.TensorDivide(wtx, engine.TensorAddScalar(wtwh, eps)));

            // Update W: W = W .* (XH') ./ (WHH' + eps)
            var ht = engine.TensorTranspose(h);
            var xht = engine.TensorMatMul(x, ht);
            var whht = engine.TensorMatMul(w, engine.TensorMatMul(h, ht));
            w = engine.TensorMultiply(w, engine.TensorDivide(xht, engine.TensorAddScalar(whht, eps)));
        }

        // Feature scores = sum of squared H values across components
        var scores = engine.ReduceSum(engine.TensorSquare(h), new[] { 0 }, false);
        _featureScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NMFBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("NMFBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NMFBasedSelector has not been fitted.");

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
