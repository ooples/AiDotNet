using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Spectral;

/// <summary>
/// Eigenvalue Ratio based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the ratio of top eigenvalues when features are
/// added, measuring how much each feature contributes to data variance.
/// </para>
/// <para><b>For Beginners:</b> Eigenvalues measure how much variance each
/// direction in data captures. This selector keeps features that contribute
/// most to the dominant directions of variance in your data.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class EigenvalueRatioSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _varianceContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? VarianceContributions => _varianceContributions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public EigenvalueRatioSelector(
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
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        var centered = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));

        // Sample covariance matrix; its diagonal is each feature's variance contribution.
        var covariance = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), NumOps.FromDouble(n - 1));
        var variances = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorMultiply(centered, centered), new[] { 0 }, false), NumOps.FromDouble(n - 1));
        T totalVariance = engine.TensorSum(variances);

        // Features with unique information (low covariance with the others) get a bonus.
        var covSum = engine.TensorSubtract(
            engine.ReduceSum(engine.TensorAbs(covariance), new[] { 1 }, false), engine.TensorAbs(variances));
        var bonus = engine.TensorDivide(variances,
            engine.TensorAddScalar(engine.TensorDivideScalar(covSum, NumOps.FromDouble(p)), NumOps.One));
        var contributions = engine.TensorAdd(variances, bonus);

        // Normalize
        if (NumOps.GreaterThan(totalVariance, NumOps.Zero))
            contributions = engine.TensorDivideScalar(contributions, totalVariance);

        _varianceContributions = contributions.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(contributions, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
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
            throw new InvalidOperationException("EigenvalueRatioSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("EigenvalueRatioSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EigenvalueRatioSelector has not been fitted.");

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
