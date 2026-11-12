using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Classification;

/// <summary>
/// F-statistic based feature selection for classification (ANOVA F-test).
/// </summary>
/// <remarks>
/// <para>
/// F-Classification uses ANOVA F-test to score features based on their ability
/// to discriminate between class means. Features with high F-scores have
/// significantly different means across classes.
/// </para>
/// <para><b>For Beginners:</b> The F-test checks if the average values of a feature
/// are different across classes. If class A has very different values than class B
/// for a feature, that feature gets a high score because it helps tell the classes apart.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FClassification<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _fScores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FScores => _fScores;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FClassification(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FClassification requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Group samples by class (rounded target values)
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length; // Number of classes
        var membership = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);

        // Between-group and within-group sums of squares
        var counts = engine.ReduceSum(membership, new[] { 0 }, false);
        var countColumn = engine.TensorBroadcastTo(engine.Reshape(counts, new[] { k, 1 }), new[] { k, p });
        var groupMeans = engine.TensorDivide(engine.TensorMatMul(engine.TensorTranspose(membership), X), countColumn);
        var overallMean = engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { k, p });
        var ssBetween = engine.ReduceSum(engine.TensorMultiply(
            engine.TensorSquare(engine.TensorSubtract(groupMeans, overallMean)), countColumn), new[] { 0 }, false);
        var ssWithin = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(X,
            engine.TensorIndexSelect(groupMeans, new Tensor<int>(classIndex, new[] { n }), 0))), new[] { 0 }, false);

        // Compute F-statistic (0 when the within-group mean square is not above 1e-10)
        int dfBetween = k - 1;
        int dfWithin = n - k;
        var zeros = new Tensor<T>(new[] { p });
        var msBetween = dfBetween > 0 ? engine.TensorDivideScalar(ssBetween, NumOps.FromDouble(dfBetween)) : zeros;
        var msWithin = dfWithin > 0 ? engine.TensorDivideScalar(ssWithin, NumOps.FromDouble(dfWithin)) : zeros;
        var usable = engine.TensorGreaterThan(msWithin, NumOps.FromDouble(1e-10));
        var fScores = engine.TensorWhere(usable, engine.TensorDivide(msBetween,
            engine.TensorWhere(usable, msWithin, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One))), zeros);

        _fScores = fScores.ToVector();
        _pValues = FeatureSelectionTensorOps.FSurvival(fScores, dfBetween, dfWithin).ToVector();

        // Select top features by F-score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(fScores, numToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("FClassification has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FClassification does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FClassification has not been fitted.");

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
