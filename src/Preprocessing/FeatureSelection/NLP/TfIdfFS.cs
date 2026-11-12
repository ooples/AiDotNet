using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.NLP;

/// <summary>
/// TF-IDF based feature selection for text data.
/// </summary>
/// <remarks>
/// <para>
/// TF-IDF (Term Frequency-Inverse Document Frequency) based selection identifies
/// features (terms) that are both frequent within documents and discriminative
/// across the corpus. Features with high TF-IDF scores are informative for
/// distinguishing documents.
/// </para>
/// <para><b>For Beginners:</b> In text data, some words appear often but are
/// meaningless (like "the"), while others are rare but important. TF-IDF balances
/// term frequency (how often a word appears in a document) with inverse document
/// frequency (how rare it is across all documents). Words that are common in some
/// documents but rare overall get high scores.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("A Statistical Interpretation of Term Specificity and Its Application in Retrieval", "https://doi.org/10.1108/eb026526", Year = 1972, Authors = "Karen Spärck Jones")]
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TfIdfFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minDf;
    private readonly double _maxDf;

    private Vector<T>? _tfidfScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MinDf => _minDf;
    public double MaxDf => _maxDf;
    public Vector<T>? TfIdfScores => _tfidfScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TfIdfFS(
        int nFeaturesToSelect = 10,
        double minDf = 0.0,
        double maxDf = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDf < 0 || minDf > 1)
            throw new ArgumentException("Min document frequency must be between 0 and 1.", nameof(minDf));
        if (maxDf < 0 || maxDf > 1)
            throw new ArgumentException("Max document frequency must be between 0 and 1.", nameof(maxDf));
        if (minDf > maxDf)
            throw new ArgumentException("Min document frequency cannot exceed max document frequency.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDf = minDf;
        _maxDf = maxDf;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows; // Number of documents
        int p = data.Columns; // Number of terms

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Document frequency of each term
        var docFreq = engine.ReduceSum(engine.TensorGreaterThan(x, NumOps.Zero), new[] { 0 }, false);

        // IDF = ln(n / df) + 1 (scikit-learn's TfidfTransformer with smooth_idf=False); 0 for unseen terms
        var seen = engine.TensorGreaterThan(docFreq, NumOps.Zero);
        var safeDocFreq = engine.TensorWhere(seen, docFreq, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
        var idf = engine.TensorWhere(
            seen,
            engine.TensorAddScalar(engine.TensorLog(engine.TensorDivide(FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.FromDouble(n)), safeDocFreq)), NumOps.One),
            new Tensor<T>(new[] { p }));

        // Average TF-IDF of each term over the documents; terms whose document-frequency ratio lies outside
        // [minDf, maxDf] score 0
        var docFreqRatio = engine.TensorDivideScalar(docFreq, NumOps.FromDouble(n));
        var tooRare = engine.TensorLessThan(docFreqRatio, NumOps.FromDouble(_minDf));
        var tooCommon = engine.TensorGreaterThan(docFreqRatio, NumOps.FromDouble(_maxDf));
        var kept = engine.TensorMultiply(FeatureSelectionTensorOps.OneMinus(tooRare), FeatureSelectionTensorOps.OneMinus(tooCommon));
        var meanTf = engine.ReduceMean(x, new[] { 0 }, false);
        var scores = engine.TensorWhere(kept, engine.TensorMultiply(meanTf, idf), new Tensor<T>(new[] { p }));
        _tfidfScores = scores.ToVector();

        // Select top features by TF-IDF score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TfIdfFS has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TfIdfFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TfIdfFS has not been fitted.");

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
