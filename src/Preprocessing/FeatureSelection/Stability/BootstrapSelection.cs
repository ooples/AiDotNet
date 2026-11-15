using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Bootstrap-based Feature Selection for robust feature importance estimation.
/// </summary>
/// <remarks>
/// <para>
/// Uses bootstrap resampling (sampling with replacement) to estimate the
/// variability of feature importance scores. Features with consistently
/// high importance across bootstrap samples are selected.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap is like asking the same question many
/// times to slightly different versions of your data (by random sampling with
/// replacement). This helps you understand which features are reliably important
/// and which ones might just be important by chance in one particular sample.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BootstrapSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _confidenceLevel;
    private readonly int? _randomState;

    private Vector<T>? _meanImportances;
    private Vector<T>? _stdImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBootstraps => _nBootstraps;
    public double ConfidenceLevel => _confidenceLevel;
    public Vector<T>? MeanImportances => _meanImportances;
    public Vector<T>? StdImportances => _stdImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BootstrapSelection(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double confidenceLevel = 0.95,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBootstraps < 1)
            throw new ArgumentException("Number of bootstraps must be at least 1.", nameof(nBootstraps));
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentException("Confidence level must be between 0 and 1.", nameof(confidenceLevel));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _confidenceLevel = confidenceLevel;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var allImportances = new Tensor<T>[_nBootstraps];

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Bootstrap sample (with replacement)
            var bootstrapIndices = new int[n];
            for (int i = 0; i < n; i++)
                bootstrapIndices[i] = random.Next(n);

            // Importance scores on the bootstrap sample: |Pearson r| (0 for a constant column or target)
            var rows = new Tensor<int>(bootstrapIndices, new[] { n });
            allImportances[b] = Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
                Engine.TensorIndexSelect(x, rows, 0), Engine.TensorIndexSelect(y, rows, 0)));
        }

        // Compute mean and (population) std of importances
        var stacked = Engine.TensorStack(allImportances, 0);
        var mean = Engine.ReduceMean(stacked, new[] { 0 }, false);
        var std = Engine.TensorSqrt(SelectorTensorOps.ColumnVariance(stacked));
        _meanImportances = mean.ToVector();
        _stdImportances = std.ToVector();

        // Compute lower confidence bound (mean - z*std/sqrt(B))
        double zScore = NormalQuantile(_confidenceLevel);
        var confidenceScores = Engine.TensorSubtract(mean,
            Engine.TensorMultiplyScalar(std, NumOps.FromDouble(zScore / Math.Sqrt(_nBootstraps))));

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(confidenceScores, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    private double NormalQuantile(double p)
    {
        // Approximation of inverse normal CDF
        double[] a = { -3.969683028665376e+01, 2.209460984245205e+02,
                       -2.759285104469687e+02, 1.383577518672690e+02,
                       -3.066479806614716e+01, 2.506628277459239e+00 };
        double[] b = { -5.447609879822406e+01, 1.615858368580409e+02,
                       -1.556989798598866e+02, 6.680131188771972e+01,
                       -1.328068155288572e+01 };
        double[] c = { -7.784894002430293e-03, -3.223964580411365e-01,
                       -2.400758277161838e+00, -2.549732539343734e+00,
                        4.374664141464968e+00, 2.938163982698783e+00 };
        double[] d = { 7.784695709041462e-03, 3.224671290700398e-01,
                       2.445134137142996e+00, 3.754408661907416e+00 };

        double pLow = 0.02425;
        double pHigh = 1 - pLow;
        double q, r;

        if (p < pLow)
        {
            q = Math.Sqrt(-2 * Math.Log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        else if (p <= pHigh)
        {
            q = p - 0.5;
            r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }
        else
        {
            q = Math.Sqrt(-2 * Math.Log(1 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
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
            throw new InvalidOperationException("BootstrapSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BootstrapSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapSelection has not been fitted.");

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
