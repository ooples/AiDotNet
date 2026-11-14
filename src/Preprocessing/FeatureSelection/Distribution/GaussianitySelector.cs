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
        int p = data.Columns;

        var engine = Engine;
        var jb = JarqueBera(Tensor<T>.FromMatrix(data));
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

    /// <summary>
    /// Jarque-Bera statistic of every column, JB = n/6 * (S² + K²/4) from population moments; lower is more
    /// Gaussian and a constant column gets the maximum value. Half widens to double for the moments, since
    /// squared deviations pass 65504 for representable data, and only the statistic is cast back.
    /// </summary>
    internal static Tensor<T> JarqueBera(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (PreprocessingTensorOps.IsNarrowFloat<T>())
        {
            var wideEngine = AiDotNetEngine.Current;
            var wide = GaussianitySelector<double>.JarqueBera(x.Cast<double>());
            // Constant columns carry double.MaxValue and map to T's maximum; every real statistic stays below that
            // maximum so the constant-column sentinel remains distinguishable after the cast.
            var constantColumns = PreprocessingTensorOps.Not(wideEngine.TensorLessThan(wide, double.MaxValue)).Cast<T>();
            var below = wideEngine.TensorMin(wide, ops.ToDouble(ops.MaxValue) * (1 - 1.0 / 64)).Cast<T>();
            return wideEngine.TensorWhere(
                constantColumns, PreprocessingTensorOps.Filled(new[] { wide.Length }, ops.MaxValue), below);
        }

        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        var shape = new[] { n, p };

        // Population moments of every feature
        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 0 }, true), shape));
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, true));
        var z = engine.TensorDivide(centered, engine.TensorBroadcastTo(std, shape));
        var zSquared = engine.TensorSquare(z);

        // Compute skewness
        var skewness = engine.ReduceMean(engine.TensorMultiply(zSquared, z), new[] { 0 }, false);

        // Compute excess kurtosis
        var kurtosis = engine.TensorSubtractScalar(
            engine.ReduceMean(engine.TensorSquare(zSquared), new[] { 0 }, false), ops.FromDouble(3));

        // Jarque-Bera test statistic: JB = n/6 * (S² + K²/4)
        // Lower JB = more Gaussian; a constant feature gets the maximum value (very non-normal)
        var jb = engine.TensorMultiplyScalar(
            engine.TensorAdd(engine.TensorSquare(skewness), engine.TensorDivideScalar(engine.TensorSquare(kurtosis), ops.FromDouble(4.0))),
            ops.FromDouble(n / 6.0));
        var constant = engine.TensorLessThan(engine.Reshape(std, new[] { p }), PreprocessingTensorOps.Tolerance<T>(1e-10));
        return engine.TensorWhere(constant, PreprocessingTensorOps.Filled(new[] { p }, ops.MaxValue), jb);
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
