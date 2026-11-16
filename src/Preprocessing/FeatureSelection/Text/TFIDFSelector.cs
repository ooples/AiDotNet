using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Text;

/// <summary>
/// TF-IDF based feature selection for text/document data.
/// </summary>
/// <remarks>
/// <para>
/// Uses Term Frequency-Inverse Document Frequency to select terms that are
/// important in some documents but not ubiquitous across all documents.
/// </para>
/// <para><b>For Beginners:</b> TF-IDF finds words that are frequent in specific
/// documents but rare overall. These terms help distinguish documents and
/// are better features than common words appearing everywhere.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TFIDFSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minDF;
    private readonly double _maxDF;

    private Vector<T>? _idfScores;
    private Vector<T>? _maxTFIDF;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? IDFScores => _idfScores;
    public Vector<T>? MaxTFIDF => _maxTFIDF;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TFIDFSelector(
        int nFeaturesToSelect = 100,
        double minDF = 0.01,
        double maxDF = 0.95,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDF < 0 || minDF > 1)
            throw new ArgumentException("minDF must be between 0 and 1.", nameof(minDF));
        if (maxDF < minDF || maxDF > 1)
            throw new ArgumentException("maxDF must be between minDF and 1.", nameof(maxDF));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDF = minDF;
        _maxDF = maxDF;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;  // Documents
        int p = data.Columns;  // Terms

        // Compute document frequency for each term
        var x = Tensor<T>.FromMatrix(data);
        // (counted as int: a 0/1 sum in T stops counting at 2048 for Half)
        var documentFrequency = Engine.ReduceSum(
            Engine.TensorGreaterThan(x, NumOps.Zero).Cast<int>(), new[] { 0 }, false).Cast<T>();

        // Filter by document frequency
        var df = Engine.TensorDivideScalar(documentFrequency, NumOps.FromDouble(n));
        var kept = Engine.TensorMultiply(
            PreprocessingTensorOps.AtLeast(df, NumOps.FromDouble(_minDF)),
            PreprocessingTensorOps.Not(Engine.TensorGreaterThan(df, NumOps.FromDouble(_maxDF))));

        // IDF with smoothing: ln((n + 1) / (df + 1)) + 1
        T one = NumOps.One;
        var idf = Engine.TensorAddScalar(Engine.TensorLog(Engine.TensorDivide(
            PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(n + 1.0)),
            Engine.TensorAddScalar(documentFrequency, one))), one);
        idf = SelectorTensorOps.ZeroUnless(kept, idf);

        // Max TF-IDF across all documents (at least 0)
        var maxTfidf = n == 0
            ? PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero)
            : Engine.TensorMax(PreprocessingTensorOps.ColumnMax(
                Engine.TensorMultiply(x, Engine.TensorBroadcastTo(Engine.Reshape(idf, new[] { 1, p }), new[] { n, p }))), NumOps.Zero);
        maxTfidf = SelectorTensorOps.ZeroUnless(kept, maxTfidf);
        _idfScores = idf.ToVector();
        _maxTFIDF = maxTfidf.ToVector();

        // Select features by max TF-IDF score (positive scores only)
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        int positive = PreprocessingTensorOps.CountNonZero(Engine.TensorGreaterThan(maxTfidf, NumOps.Zero));
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(maxTfidf, Math.Min(nToSelect, positive));

        if (_selectedIndices.Length == 0)
        {
            // Fallback: select highest IDF terms
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(idf, nToSelect);
        }
        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TFIDFSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TFIDFSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TFIDFSelector has not been fitted.");

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
