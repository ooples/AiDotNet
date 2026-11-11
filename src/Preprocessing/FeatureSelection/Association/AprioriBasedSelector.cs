using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Association;

/// <summary>
/// Apriori-based Association Rule Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses association rule mining concepts to find features that frequently
/// co-occur with the target class, selecting features with high support
/// and confidence.
/// </para>
/// <para><b>For Beginners:</b> Association rules find patterns like "people who
/// buy bread often buy butter." We apply this to features: features that
/// "associate" strongly with the target class (high confidence) and appear
/// often (high support) are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AprioriBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minSupport;
    private readonly double _minConfidence;
    private readonly int _nBins;

    private Vector<T>? _associationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? AssociationScores => _associationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AprioriBasedSelector(
        int nFeaturesToSelect = 10,
        double minSupport = 0.1,
        double minConfidence = 0.5,
        int nBins = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minSupport = minSupport;
        _minConfidence = minConfidence;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AprioriBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var positive = engine.Reshape(
            engine.TensorGreaterThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5)), new[] { 1, n }); // Binarize

        // Discretize features: bin = min(floor((x - min) / (max - min + 1e-10) * nBins), nBins - 1)
        var min = engine.Reshape(PreprocessingTensorOps.ColumnMin(X), new[] { 1, p });
        var range = engine.TensorAddScalar(
            engine.TensorSubtract(engine.Reshape(PreprocessingTensorOps.ColumnMax(X), new[] { 1, p }), min), NumOps.FromDouble(1e-10));
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(X, min), range), NumOps.FromDouble(_nBins));
        var bins = engine.TensorMin(engine.TensorFloor(scaled), NumOps.FromDouble(_nBins - 1));
        var oneHot = PreprocessingTensorOps.OneHotBins(bins, _nBins);

        T count = NumOps.FromDouble(n);
        int nPositive = PreprocessingTensorOps.CountNonZero(positive);
        double targetSupport = (double)nPositive / n;

        // Per (feature, bin): transactions with the value and those that are also positive
        var featureCount = engine.ReduceSum(oneHot, new[] { 0 }, false);
        var jointCount = engine.Reshape(engine.TensorMatMul(positive, oneHot), new[] { p * _nBins });

        var support = engine.TensorDivideScalar(jointCount, count);
        var confidence = engine.TensorDivide(jointCount, featureCount);
        var lift = engine.TensorDivideScalar(confidence, NumOps.FromDouble(targetSupport + 1e-10));

        // Score combines support, confidence, and lift for the rules that clear both minimums
        var score = engine.TensorMultiply(engine.TensorMultiply(support, confidence), lift);
        var qualifies = engine.TensorMultiply(
            engine.TensorMultiply(
                engine.TensorGreaterThan(featureCount, NumOps.Zero),
                PreprocessingTensorOps.AtLeast(support, NumOps.FromDouble(_minSupport))),
            PreprocessingTensorOps.AtLeast(confidence, NumOps.FromDouble(_minConfidence)));
        var masked = engine.TensorWhere(qualifies, score, new Tensor<T>(new[] { p * _nBins }));
        var scores = engine.TensorMax(
            engine.ReduceMax(engine.Reshape(masked, new[] { p, _nBins }), new[] { 1 }, false), NumOps.Zero);
        _associationScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AprioriBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AprioriBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AprioriBasedSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
