using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.DomainSpecific;

/// <summary>
/// Feature selection for text/NLP data.
/// </summary>
/// <remarks>
/// <para>
/// TextFeatureSelector is designed for selecting features from text data,
/// such as word frequencies, TF-IDF scores, or word embeddings. It considers
/// text-specific properties like term frequency and document frequency.
/// </para>
/// <para><b>For Beginners:</b> Text data has unique properties - some words are
/// common but uninformative (like "the"), while rare words might be very
/// distinctive. This selector picks words/features that best distinguish
/// between different text categories.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TextFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minDocumentFrequency;
    private readonly double _maxDocumentFrequency;
    private readonly bool _useChiSquare;

    private Vector<T>? _featureScores;
    private Vector<T>? _documentFrequencies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MinDocumentFrequency => _minDocumentFrequency;
    public double MaxDocumentFrequency => _maxDocumentFrequency;
    public bool UseChiSquare => _useChiSquare;
    public Vector<T>? FeatureScores => _featureScores;
    public Vector<T>? DocumentFrequencies => _documentFrequencies;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TextFeatureSelector(
        int nFeaturesToSelect = 10,
        double minDocumentFrequency = 0.01,
        double maxDocumentFrequency = 0.95,
        bool useChiSquare = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDocumentFrequency < 0 || minDocumentFrequency > 1)
            throw new ArgumentException("Min document frequency must be between 0 and 1.", nameof(minDocumentFrequency));
        if (maxDocumentFrequency < 0 || maxDocumentFrequency > 1)
            throw new ArgumentException("Max document frequency must be between 0 and 1.", nameof(maxDocumentFrequency));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDocumentFrequency = minDocumentFrequency;
        _maxDocumentFrequency = maxDocumentFrequency;
        _useChiSquare = useChiSquare;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TextFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (data.Rows == 0 || data.Columns == 0)
            throw new ArgumentException("Data must contain at least one row and one column.", nameof(data));

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Compute document frequencies (counted in int: a Half sum of the 0/1 mask stops at 2048)
        var present = engine.TensorGreaterThan(X, NumOps.Zero);
        var documentCounts = engine.ReduceSum(present.Cast<int>(), new[] { 0 }, false).Cast<double>();
        var documentFrequencies = engine.TensorDivideScalar(documentCounts, (double)n).Cast<T>();
        _documentFrequencies = documentFrequencies.ToVector();

        // Filter by document frequency
        var candidateMask = engine.TensorMultiply(
            PreprocessingTensorOps.AtLeast(documentFrequencies, NumOps.FromDouble(_minDocumentFrequency)),
            PreprocessingTensorOps.Not(engine.TensorGreaterThan(documentFrequencies, NumOps.FromDouble(_maxDocumentFrequency))));
        var candidates = PreprocessingTensorOps.MaskPositions(candidateMask);
        var candidateSet = new HashSet<int>(candidates);

        // Compute feature scores (0 for features outside the document-frequency band)
        var scores = _useChiSquare
            ? ComputeChiSquareScores(present.Cast<double>(), target, n, p).Cast<T>()
            : engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, y));
        scores = engine.TensorWhere(candidateMask, scores, new Tensor<T>(new[] { p }));
        _featureScores = scores.ToVector();

        // Select top features from candidates
        var order = PreprocessingTensorOps.StableOrder(scores, descending: true);
        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Length);
        _selectedIndices = order
            .Where(j => candidateSet.Contains(j))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        // If not enough candidates, add from remaining features
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var remaining = order
                .Where(j => !candidateSet.Contains(j))
                .Take(_nFeaturesToSelect - _selectedIndices.Length);

            _selectedIndices = _selectedIndices.Concat(remaining).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    /// <summary>
    /// Chi-square statistic of every feature's presence against the classes. The contingency counts and the
    /// expected-count products exceed Half's exact-integer (2048) and finite (65504) ranges, so the table is built
    /// and scored in double.
    /// </summary>
    private static Tensor<double> ComputeChiSquareScores(Tensor<double> present, Vector<T> target, int n, int p)
    {
        var engine = AiDotNetEngine.Current;

        // Get class counts
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        var membership = PreprocessingTensorOps.ClassOneHot<double>(classIndex, k);
        var classTotals = engine.ReduceSum(membership, new[] { 0 }, false);

        // Contingency table: rows with the feature present / absent in each class ([p, k] each)
        var observedPresent = engine.TensorMatMul(engine.TensorTranspose(present), membership);
        var observedAbsent = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(classTotals, new[] { 1, k }), new[] { p, k }), observedPresent);

        // Expected counts: (feature total) * (class total) / n
        var featureCount = engine.ReduceSum(present, new[] { 0 }, false);
        double total = n;
        var expectedPresent = engine.TensorDivideScalar(engine.TensorOuter(featureCount, classTotals), total);
        var expectedAbsent = engine.TensorDivideScalar(engine.TensorOuter(
            engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { p }, total), featureCount), classTotals), total);

        // Compute chi-square statistic over the cells with a positive expected count
        return engine.TensorAdd(ChiSquareTerms(observedPresent, expectedPresent), ChiSquareTerms(observedAbsent, expectedAbsent));
    }

    private static Tensor<double> ChiSquareTerms(Tensor<double> observed, Tensor<double> expected)
    {
        var engine = AiDotNetEngine.Current;
        var terms = engine.TensorDivide(engine.TensorSquare(engine.TensorSubtract(observed, expected)), expected);
        var kept = engine.TensorWhere(engine.TensorGreaterThan(expected, 0.0), terms, new Tensor<double>(expected.Shape.ToArray()));
        return engine.ReduceSum(kept, new[] { 1 }, false);
    }
    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TextFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TextFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TextFeatureSelector has not been fitted.");

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
            return _selectedIndices.Select(i => $"Term{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
