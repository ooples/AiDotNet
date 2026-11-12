using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Distribution;

/// <summary>
/// Gaussianity (Normality) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how closely their distributions match a Gaussian
/// (normal) distribution, using the Jarque-Bera test statistic.
/// </para>
/// <para><b>For Beginners:</b> Many statistical methods assume data follows a
/// bell curve (normal distribution). This selector measures how "normal" each
/// feature is. You can select features that are most normal (for parametric methods)
/// or least normal (to find interesting non-standard patterns).
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GaussianitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferGaussian;

    private Vector<T>? _jbStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferGaussian => _preferGaussian;
    public Vector<T>? JBStatistics => _jbStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GaussianitySelector(
        int nFeaturesToSelect = 10,
        bool preferGaussian = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferGaussian = preferGaussian;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var shape = new[] { n, p };

        // Population moments of every feature
        var centered = engine.TensorSubtract(X, engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), shape));
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, true));
        var z = engine.TensorDivide(centered, engine.TensorBroadcastTo(std, shape));
        var zSquared = engine.TensorSquare(z);

        // Compute skewness
        var skewness = engine.ReduceMean(engine.TensorMultiply(zSquared, z), new[] { 0 }, false);

        // Compute excess kurtosis
        var kurtosis = engine.TensorSubtractScalar(
            engine.ReduceMean(engine.TensorSquare(zSquared), new[] { 0 }, false), NumOps.FromDouble(3));

        // Jarque-Bera test statistic: JB = n/6 * (S² + K²/4)
        // Lower JB = more Gaussian; a constant feature gets the maximum value (very non-normal)
        var jb = engine.TensorMultiplyScalar(
            engine.TensorAdd(engine.TensorSquare(skewness), engine.TensorDivideScalar(engine.TensorSquare(kurtosis), NumOps.FromDouble(4.0))),
            NumOps.FromDouble(n / 6.0));
        var constant = engine.TensorLessThan(engine.Reshape(std, new[] { p }), PreprocessingTensorOps.Tolerance<T>(1e-10));
        jb = engine.TensorWhere(constant, PreprocessingTensorOps.Filled(new[] { p }, NumOps.MaxValue), jb);
        _jbStatistics = jb.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferGaussian)
        {
            // Prefer features with low JB statistic (more Gaussian)
            _selectedIndices = PreprocessingTensorOps.BottomKIndicesSorted(jb, numToSelect);
        }
        else
        {
            // Prefer features with high JB statistic (less Gaussian), excluding constant features
            var isConstant = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
                PreprocessingTensorOps.Not(engine.TensorLessThan(jb, NumOps.MaxValue))));
            _selectedIndices = PreprocessingTensorOps.StableOrder(jb, descending: true)
                .Where(j => !isConstant.Contains(j))
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
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
            throw new InvalidOperationException("GaussianitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GaussianitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GaussianitySelector has not been fitted.");

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
