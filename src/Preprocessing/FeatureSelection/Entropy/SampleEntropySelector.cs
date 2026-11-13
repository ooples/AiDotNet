using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Sample Entropy (SampEn) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their sample entropy, an improved version of
/// approximate entropy that is less biased for short time series.
/// </para>
/// <para><b>For Beginners:</b> Sample entropy is like approximate entropy but
/// doesn't count self-matches, making it more reliable for shorter sequences.
/// It measures unpredictability - low SampEn means regular, predictable patterns;
/// high SampEn means complex, chaotic behavior.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SampleEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _embeddingDimension;
    private readonly double _tolerance;

    private Vector<T>? _sampenValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int EmbeddingDimension => _embeddingDimension;
    public double Tolerance => _tolerance;
    public Vector<T>? SampEnValues => _sampenValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SampleEntropySelector(
        int nFeaturesToSelect = 10,
        int embeddingDimension = 2,
        double tolerance = 0.2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (embeddingDimension < 1)
            throw new ArgumentException("Embedding dimension must be at least 1.", nameof(embeddingDimension));
        if (tolerance <= 0)
            throw new ArgumentException("Tolerance must be positive.", nameof(tolerance));

        _nFeaturesToSelect = nFeaturesToSelect;
        _embeddingDimension = embeddingDimension;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < _embeddingDimension + 2)
            throw new ArgumentException($"Need at least {_embeddingDimension + 2} samples.");

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Tolerance r = tolerance * population std per feature; features with r below 1e-10 score 0
        var tolerances = engine.TensorMultiplyScalar(FeatureSelectionTensorOps.ColumnStd(X), NumOps.FromDouble(_tolerance));
        var degenerate = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            engine.TensorLessThan(tolerances, PreprocessingTensorOps.Tolerance<T>(1e-10))));

        // Richman and Moorman (2000): B and A count the matching pairs among the same n - m templates
        int templates = n - _embeddingDimension;
        var bCounts = new Tensor<double>[p];
        var aCounts = new Tensor<double>[p];
        for (int j = 0; j < p; j++)
        {
            if (degenerate.Contains(j))
            {
                bCounts[j] = new Tensor<double>(new[] { 1 });
                aCounts[j] = new Tensor<double>(new[] { 1 });
                continue;
            }

            var col = PreprocessingTensorOps.ColumnVector(X, j);
            var r = engine.TensorSlice(tolerances, new[] { j }, new[] { 1 });

            // Count matches for m and m+1 (excluding self-matches)
            bCounts[j] = CountMatches(col, _embeddingDimension, templates, r);
            aCounts[j] = CountMatches(col, _embeddingDimension + 1, templates, r);
        }

        // SampEn = -ln(A/B), 0 when either count is 0. The pair counts reach (n - m)^2 / 2, past Half's exact
        // integers (2048) and its maximum (65504), so they are counted in int and the ratio is taken in double.
        var shape = new[] { p };
        var b = p == 0 ? new Tensor<double>(shape) : engine.TensorConcatenate(bCounts, 0);
        var a = p == 0 ? new Tensor<double>(shape) : engine.TensorConcatenate(aCounts, 0);
        var defined = engine.TensorMultiply(engine.TensorGreaterThan(b, 0.0), engine.TensorGreaterThan(a, 0.0));
        var ones = PreprocessingTensorOps.Filled(shape, 1.0);
        var sampen = engine.TensorWhere(defined, engine.TensorNegate(engine.TensorLog(engine.TensorDivide(
            engine.TensorWhere(defined, a, ones), engine.TensorWhere(defined, b, ones)))), new Tensor<double>(shape)).Cast<T>();
        _sampenValues = sampen.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(sampen, numToSelect);

        IsFitted = true;
    }

    /// <summary>Number of template pairs i &lt; k (of the first <paramref name="count"/> templates) within r. Returns [1].</summary>
    private Tensor<double> CountMatches(Tensor<T> data, int m, int count, Tensor<T> r)
    {
        // Every template matches itself, so the pairs are (all matches - count) / 2
        var matches = Engine.ReduceSum(Engine.Reshape(
            FeatureSelectionTensorOps.TemplateMatches(data, m, count, r).Cast<int>(), new[] { count * count }), new[] { 0 }, true);
        return Engine.TensorDivideScalar(Engine.TensorSubtractScalar(matches.Cast<double>(), (double)count), 2.0);
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
            throw new InvalidOperationException("SampleEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SampleEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SampleEntropySelector has not been fitted.");

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
