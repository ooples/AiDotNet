using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Adjusted Mutual Information correcting for chance agreement.
/// </summary>
/// <remarks>
/// <para>
/// Adjusted Mutual Information (AMI) corrects for the fact that random
/// labelings have non-zero expected MI. It accounts for the number of
/// clusters to give a more accurate measure of agreement.
/// </para>
/// <para><b>For Beginners:</b> Just by chance, two random groupings will
/// share some information. AMI subtracts this expected chance agreement
/// to give a cleaner measure of true dependency.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AdjustedMutualInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _amiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? AMIScores => _amiScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AdjustedMutualInformation(
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
            "AdjustedMutualInformation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize target (distinct values) and features (range-scaled bins, one bin for a constant feature)
        var y = Tensor<T>.FromVector(target);
        var classLabels = PreprocessingTensorOps.SortedUnique(y);
        int k = classLabels.Length;
        var classOneHot = engine.TensorOneHot<T>(PreprocessingTensorOps.LookupSorted(classLabels, y, out _), k);
        var classCounts = engine.ReduceSum(classOneHot, new[] { 0 }, false);
        var featureBins = PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins);
        var joint = FeatureSelectionTensorOps.JointBinCounts(featureBins, classOneHot, _nBins);

        // MI = H(Y) - H(Y|X), expected MI under permutation, and AMI = (MI - EMI) / (max(H(X), H(Y)) - EMI)
        T hY = FeatureSelectionTensorOps.ClassEntropyNats(classOneHot);
        var hX = FeatureSelectionTensorOps.BinEntropyNats(joint, n, out var conditionalEntropy);
        var mi = engine.TensorNegate(engine.TensorAddScalar(conditionalEntropy, NumOps.Negate(hY)));
        var emi = FeatureSelectionTensorOps.ExpectedMutualInformation(
            engine.Reshape(engine.ReduceSum(joint, new[] { 2 }, false), new[] { p, _nBins }), classCounts, n);
        var denominator = engine.TensorSubtract(engine.TensorMax(hX, hY), emi);
        var usable = engine.TensorGreaterThan(engine.TensorAbs(denominator), NumOps.FromDouble(1e-10));
        var amiScores = engine.TensorWhere(usable, engine.TensorDivide(engine.TensorSubtract(mi, emi),
            engine.TensorWhere(usable, denominator, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One))), new Tensor<T>(new[] { p }));
        _amiScores = amiScores.ToVector();

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(amiScores, nToSelect);

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
            throw new InvalidOperationException("AdjustedMutualInformation has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AdjustedMutualInformation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AdjustedMutualInformation has not been fitted.");

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
