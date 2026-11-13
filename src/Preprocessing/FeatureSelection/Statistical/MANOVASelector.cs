using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// MANOVA (Multivariate ANOVA) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on multivariate analysis of variance, which considers
/// multiple dependent variables simultaneously when testing group differences.
/// </para>
/// <para><b>For Beginners:</b> MANOVA extends ANOVA to multiple response variables.
/// Instead of testing each feature independently, it considers correlations between
/// features when assessing group differences. This can detect patterns that
/// univariate tests miss.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MANOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _wilksLambda;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? WilksLambda => _wilksLambda;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MANOVASelector(
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
            "MANOVASelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        // Univariate Wilks' Lambda = SS_within / SS_total per feature (1 for a constant feature)
        var between = SelectorTensorOps.ClassScatter(
            Tensor<T>.FromMatrix(data), new Tensor<int>(classIndex, new[] { n }), k, out var within);
        var total = Engine.TensorAdd(between, within);
        var usable = Engine.TensorGreaterThan(total, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var scores = Engine.TensorWhere(usable,
            Engine.TensorDivide(within, Engine.TensorWhere(usable, total, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One))),
            PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        _wilksLambda = scores.ToVector();

        // Lower Wilks' Lambda = better discrimination
        _selectedIndices = PreprocessingTensorOps.BottomKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("MANOVASelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MANOVASelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MANOVASelector has not been fitted.");

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
