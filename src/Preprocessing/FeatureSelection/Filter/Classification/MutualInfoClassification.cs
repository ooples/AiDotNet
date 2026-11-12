using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Classification;

/// <summary>
/// Mutual Information for classification-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Mutual Information Classification estimates the mutual information between
/// each feature and the discrete class label. MI measures how much knowing
/// the feature reduces uncertainty about the class.
/// </para>
/// <para><b>For Beginners:</b> Mutual information measures how much knowing
/// a feature's value helps predict the class. Unlike correlation, MI can detect
/// any type of relationship, not just linear ones. A high MI score means the
/// feature contains useful information for classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MutualInfoClassification<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _miScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? MIScores => _miScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MutualInfoClassification(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MutualInfoClassification requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        T count = NumOps.FromDouble(n);

        // Class labels (rounded target values) and range-scaled feature bins (0 for a constant feature)
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int nClasses = classes.Length;
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, nClasses);
        var featureBins = PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins);

        // Joint, feature and class probabilities
        var cellShape = new[] { p, _nBins, nClasses };
        var joint = FeatureSelectionTensorOps.JointBinCounts(featureBins, classOneHot, _nBins);
        var pJoint = engine.TensorDivideScalar(joint, count);
        var pFeature = engine.TensorBroadcastTo(engine.TensorDivideScalar(engine.ReduceSum(joint, new[] { 2 }, true), count), cellShape);
        var pClass = engine.TensorBroadcastTo(engine.Reshape(engine.TensorDivideScalar(
            engine.ReduceSum(classOneHot, new[] { 0 }, false), count), new[] { 1, 1, nClasses }), cellShape);

        // Mutual information (nats) over the non-empty joint cells, clamped at 0
        var occupied = engine.TensorGreaterThan(joint, NumOps.Zero);
        var ones = PreprocessingTensorOps.Filled(cellShape, NumOps.One);
        var ratio = engine.TensorDivide(pJoint, engine.TensorWhere(occupied, engine.TensorMultiply(pFeature, pClass), ones));
        var terms = engine.TensorWhere(occupied, engine.TensorMultiply(pJoint,
            engine.TensorLog(engine.TensorAddScalar(ratio, NumOps.FromDouble(1e-10)))), new Tensor<T>(cellShape));
        var miScores = engine.TensorMax(engine.ReduceSum(terms, new[] { 1, 2 }, false), NumOps.Zero);
        _miScores = miScores.ToVector();

        // Select top features by MI
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(miScores, numToSelect);

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
            throw new InvalidOperationException("MutualInfoClassification has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MutualInfoClassification does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInfoClassification has not been fitted.");

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
