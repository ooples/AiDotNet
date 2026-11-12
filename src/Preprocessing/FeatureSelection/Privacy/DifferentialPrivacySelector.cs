using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Privacy;

/// <summary>
/// Differential Privacy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features with the exponential mechanism (McSherry and Talwar 2007) so that the selected set is
/// epsilon-differentially private with respect to replacing one row of the data. The utility of feature j is the
/// absolute covariance |cov(x_j, y)| of the feature and target after clipping them to
/// [-featureBound, featureBound] and [-targetBound, targetBound]. With values in those ranges, replacing one of the
/// n rows changes the mean of x_j y by at most 2 B_x B_y / n and the product of the means by at most 4 B_x B_y / n,
/// so the utility has sensitivity Delta = 6 B_x B_y / n.
/// </para>
/// <para>
/// The k features are chosen by k rounds of the exponential mechanism, each with budget epsilon / k over the
/// features not chosen yet, selecting feature j with probability proportional to
/// exp((epsilon / k) u_j / (2 Delta)); by sequential composition the whole selection is epsilon-DP. The rounds
/// are sampled in one shot with the Gumbel-max trick: adding Gumbel noise of scale 2 k Delta / epsilon to every
/// utility and taking the k largest has the same distribution (Durfee and Rogers 2019). Only the selected indices
/// are released; the noisy utilities are not.
/// </para>
/// <para><b>For Beginners:</b> Differential privacy adds calibrated randomness to the feature choice so that
/// whether any single individual is in the dataset barely changes which features get selected. Scale the
/// features and target to known bounds (for example from their documented domains, not from the data itself)
/// and pass those bounds; values outside them are clipped so the privacy guarantee always holds.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DifferentialPrivacySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _epsilon;
    private readonly int? _randomState;
    private readonly double _featureBound;
    private readonly double _targetBound;

    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _noiseScale;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>The total privacy budget of the selection.</summary>
    public double Epsilon => _epsilon;

    /// <summary>Features are clipped to [-FeatureBound, FeatureBound] before the utilities are computed.</summary>
    public double FeatureBound => _featureBound;

    /// <summary>The target is clipped to [-TargetBound, TargetBound] before the utilities are computed.</summary>
    public double TargetBound => _targetBound;

    /// <summary>The Gumbel noise scale 2 k Delta / epsilon used by the last fit (0 before fitting).</summary>
    public double NoiseScale => _noiseScale;

    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;

    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DifferentialPrivacySelector(
        int nFeaturesToSelect = 10,
        double epsilon = 1.0,
        int? randomState = null,
        int[]? columnIndices = null,
        double featureBound = 1.0,
        double targetBound = 1.0)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (epsilon <= 0)
            throw new ArgumentException("Epsilon must be positive.", nameof(epsilon));
        if (featureBound <= 0)
            throw new ArgumentException("Feature bound must be positive.", nameof(featureBound));
        if (targetBound <= 0)
            throw new ArgumentException("Target bound must be positive.", nameof(targetBound));

        _nFeaturesToSelect = nFeaturesToSelect;
        _epsilon = epsilon;
        _randomState = randomState;
        _featureBound = featureBound;
        _targetBound = targetBound;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DifferentialPrivacySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var utility = Utilities(data, target, _featureBound, _targetBound);
        _noiseScale = NoiseScaleFor(_epsilon, numToSelect, n, _featureBound, _targetBound);
        _selectedIndices = ExponentialMechanismTopK(utility, numToSelect, _noiseScale, rand);

        IsFitted = true;
    }

    /// <summary>
    /// The utilities |cov(clip(x_j), clip(y))| of every feature, with features clipped to
    /// [-<paramref name="featureBound"/>, <paramref name="featureBound"/>] and the target to
    /// [-<paramref name="targetBound"/>, <paramref name="targetBound"/>]. Returns [p].
    /// </summary>
    internal static Tensor<T> Utilities(Matrix<T> data, Vector<T> target, double featureBound, double targetBound)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int p = data.Columns;
        var x = engine.TensorClamp(Tensor<T>.FromMatrix(data), ops.FromDouble(-featureBound), ops.FromDouble(featureBound));
        var y = engine.TensorClamp(Tensor<T>.FromVector(target), ops.FromDouble(-targetBound), ops.FromDouble(targetBound));
        var yColumn = engine.TensorBroadcastTo(engine.Reshape(y, new[] { n, 1 }), new[] { n, p });
        var meanProduct = engine.ReduceMean(engine.TensorMultiply(x, yColumn), new[] { 0 }, false);
        var meanX = engine.ReduceMean(x, new[] { 0 }, false);
        var meanY = engine.TensorBroadcastTo(engine.ReduceMean(y, new[] { 0 }, true), new[] { p });
        return engine.TensorAbs(engine.TensorSubtract(meanProduct, engine.TensorMultiply(meanX, meanY)));
    }

    /// <summary>
    /// The Gumbel noise scale 2 k Delta / epsilon of k exponential-mechanism rounds with budget epsilon / k each,
    /// for the utility sensitivity Delta = 6 B_x B_y / n.
    /// </summary>
    internal static double NoiseScaleFor(double epsilon, int k, int n, double featureBound, double targetBound)
    {
        double sensitivity = 6.0 * featureBound * targetBound / n;
        return 2.0 * k * sensitivity / epsilon;
    }

    /// <summary>
    /// The k features of the peeling exponential mechanism, sampled in one shot: the indices (ascending) of the
    /// k largest u_j + scale * G_j with G_j = -log(-log U_j) standard Gumbel, U_j uniform.
    /// </summary>
    internal static int[] ExponentialMechanismTopK(Tensor<T> utility, int k, double scale, Random rand)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var uniform = PopulationSearchOps.Uniform<T>(rand, utility.Length);
        var gumbel = engine.TensorNegate(engine.TensorLog(engine.TensorNegate(engine.TensorLog(uniform))));
        var noisy = engine.TensorAdd(utility, engine.TensorMultiplyScalar(gumbel, ops.FromDouble(scale)));
        return FeatureSelectionTensorOps.TopKIndicesSorted(noisy, k);
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
            throw new InvalidOperationException("DifferentialPrivacySelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DifferentialPrivacySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialPrivacySelector has not been fitted.");

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
