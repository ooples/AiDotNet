using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.NLP;

/// <summary>
/// Document Frequency based feature selection for text data.
/// </summary>
/// <remarks>
/// <para>
/// Document Frequency (DF) selection filters terms based on how many documents
/// they appear in. Terms appearing in too few or too many documents are removed,
/// as they are either too rare to be useful or too common to be discriminative.
/// </para>
/// <para><b>For Beginners:</b> Some words appear in almost every document (like "is" or "the")
/// and don't help distinguish documents. Other words appear only once or twice in the
/// entire corpus and might be typos or too specific. DF filtering keeps words that appear
/// in a reasonable number of documents - common enough to be meaningful but not so common
/// that everyone uses them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Foundations of Statistical Natural Language Processing", "https://nlp.stanford.edu/fsnlp/", Year = 1999, Authors = "Christopher D. Manning, Hinrich Schütze")]
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DocumentFrequencyFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _minDf;
    private readonly int _maxDf;

    private int[]? _documentFrequencies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MinDf => _minDf;
    public int MaxDf => _maxDf;
    public int[]? DocumentFrequencies => _documentFrequencies;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DocumentFrequencyFS(
        int nFeaturesToSelect = 10,
        int minDf = 1,
        int maxDf = int.MaxValue,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDf < 0)
            throw new ArgumentException("Min document frequency must be non-negative.", nameof(minDf));
        if (maxDf < minDf)
            throw new ArgumentException("Max document frequency cannot be less than min.", nameof(maxDf));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDf = minDf;
        _maxDf = maxDf;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Document frequency of each term: the number of documents (rows) where it occurs (value > 0)
        var engine = Engine;
        var occurs = engine.TensorGreaterThan(Tensor<T>.FromMatrix(data), NumOps.Zero);
        var df = engine.ReduceSum(occurs, new[] { 0 }, false);
        _documentFrequencies = df.Cast<int>().AsSpan().ToArray();

        // Filter by document frequency and select the most frequent terms
        var atLeastMin = PreprocessingTensorOps.OneMinus(engine.TensorLessThan(df, NumOps.FromDouble(_minDf)));
        var atMostMax = PreprocessingTensorOps.OneMinus(engine.TensorGreaterThan(df, NumOps.FromDouble(_maxDf)));
        var inRange = engine.TensorMultiply(atLeastMin, atMostMax);

        if (PreprocessingTensorOps.CountNonZero(inRange) == 0 && p > 0)
        {
            // No features pass filter, select features closest to criteria
            _selectedIndices = FeatureSelectionTensorOps.TopKWhere(df, atLeastMin, Math.Min(_nFeaturesToSelect, p));

            if (_selectedIndices.Length == 0)
                _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(df, Math.Min(_nFeaturesToSelect, p));
        }
        else
        {
            _selectedIndices = FeatureSelectionTensorOps.TopKWhere(df, inRange, _nFeaturesToSelect);
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DocumentFrequencyFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DocumentFrequencyFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DocumentFrequencyFS has not been fitted.");

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
