using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Point-Biserial Correlation for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Point-biserial correlation measures the relationship between a continuous
/// variable and a dichotomous (binary) variable. It's the appropriate correlation
/// measure when the target is binary.
/// </para>
/// <para><b>For Beginners:</b> When you're trying to predict something that has
/// only two outcomes (like yes/no, pass/fail), this measures how well a continuous
/// feature separates the two groups. Higher absolute values mean the feature is
/// better at distinguishing between the classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PointBiserial<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Correlations => _correlations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PointBiserial(
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
            "PointBiserial requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Split samples by class (target < 0.5 is class 0)
        var class1 = PreprocessingTensorOps.AtLeast(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5));
        int n1 = PreprocessingTensorOps.CountNonZero(class1);
        int n0 = n - n1;

        if (n0 < 2 || n1 < 2)
        {
            _correlations = new Vector<T>(p);
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        // r_pb = |mean1 - mean0| / std_n * sqrt(n0 n1 / n^2) (0 for a feature whose std is below 1e-10)
        // Class means through GroupMeans, which sums in double: a Half class sum stops growing once it passes 2048
        // times the values' spacing and overflows at 65504
        var classMeans = FeatureSelectionTensorOps.GroupMeans(
            PreprocessingTensorOps.ClassOneHot<T>(class1.Cast<int>().AsSpan().ToArray(), 2), X, out _);
        var mean0 = engine.Reshape(PreprocessingTensorOps.Row(classMeans, 0), new[] { p });
        var mean1 = engine.Reshape(PreprocessingTensorOps.Row(classMeans, 1), new[] { p });
        var stdTotal = FeatureSelectionTensorOps.ColumnStd(X);
        var usable = PreprocessingTensorOps.AtLeast(stdTotal, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var safeStd = engine.TensorWhere(usable, stdTotal, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        T sqrtPq = NumOps.Sqrt(NumOps.FromDouble((double)n0 * n1 / ((double)n * n)));
        var correlations = engine.TensorWhere(usable, engine.TensorAbs(engine.TensorMultiplyScalar(
            engine.TensorDivide(engine.TensorSubtract(mean1, mean0), safeStd), sqrtPq)), new Tensor<T>(new[] { p }));
        _correlations = correlations.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(correlations, numToSelect);

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
            throw new InvalidOperationException("PointBiserial has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PointBiserial does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PointBiserial has not been fitted.");

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
