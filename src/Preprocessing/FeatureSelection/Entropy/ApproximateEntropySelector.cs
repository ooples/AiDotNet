using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Approximate Entropy (ApEn) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their approximate entropy, which quantifies
/// the unpredictability and complexity of time series data.
/// </para>
/// <para><b>For Beginners:</b> Approximate entropy measures how likely it is
/// that similar patterns in a sequence remain similar when extended by one point.
/// Low ApEn means the sequence is predictable (patterns repeat); high ApEn means
/// it's complex and hard to predict. Often used for physiological signals.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ApproximateEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _embeddingDimension;
    private readonly double _tolerance;

    private Vector<T>? _apenValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int EmbeddingDimension => _embeddingDimension;
    public double Tolerance => _tolerance;
    public Vector<T>? ApEnValues => _apenValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ApproximateEntropySelector(
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
        var tolerances = engine.TensorMultiplyScalar(FeatureScoringTensorOps.ColumnStd(X), NumOps.FromDouble(_tolerance));
        var degenerate = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            engine.TensorLessThan(tolerances, PreprocessingTensorOps.Tolerance<T>(1e-10))));

        var apenValues = new Tensor<double>[p];
        for (int j = 0; j < p; j++)
        {
            if (degenerate.Contains(j))
            {
                apenValues[j] = new Tensor<double>(new[] { 1 });
                continue;
            }

            var col = PreprocessingTensorOps.ColumnVector(X, j);
            var r = engine.TensorSlice(tolerances, new[] { j }, new[] { 1 });

            // ApEn = phi(m) - phi(m+1)
            apenValues[j] = engine.TensorSubtract(
                ComputePhi(col, _embeddingDimension, r), ComputePhi(col, _embeddingDimension + 1, r));
        }

        var apen = (p == 0 ? new Tensor<double>(new[] { 0 }) : engine.TensorConcatenate(apenValues, 0)).Cast<T>();
        _apenValues = apen.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(apen, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// phi(m) = mean over the n - m + 1 templates of ln(C_i), C_i the fraction of templates (self included) within
    /// r of template i (Pincus 1991). Returns [1].
    /// </summary>
    private Tensor<double> ComputePhi(Tensor<T> data, int m, Tensor<T> r)
    {
        int n = data.Length;
        int count = n - m + 1;
        if (count <= 0) return new Tensor<double>(new[] { 1 });

        // Counted in int (a Half sum of the 0/1 matches stops at 2048); the fractions and logs are taken in double
        // Counted per template row in bounded blocks, not from a [count, count] match matrix
        var matches = FeatureScoringTensorOps.TemplateMatchCounts(data, m, count, r);
        var logs = Engine.TensorLog(Engine.TensorDivideScalar(matches.Cast<double>(), (double)count));
        return Engine.ReduceMean(logs, new[] { 0 }, true);
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
            throw new InvalidOperationException("ApproximateEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ApproximateEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ApproximateEntropySelector has not been fitted.");

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
