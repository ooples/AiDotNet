using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Huber Correlation-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Huber M-estimator for robust correlation computation, downweighting
/// the influence of outliers in the feature-target relationship.
/// </para>
/// <para><b>For Beginners:</b> Regular correlation can be strongly affected by
/// outliers. The Huber method treats moderate deviations normally but reduces
/// the influence of extreme values. This gives you feature-target correlations
/// that better reflect the typical relationship, not just outlier effects.
/// </para>
/// </remarks>
public class HuberCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _huberDelta;

    private double[]? _robustCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double HuberDelta => _huberDelta;
    public double[]? RobustCorrelations => _robustCorrelations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HuberCorrelationSelector(
        int nFeaturesToSelect = 10,
        double huberDelta = 1.35,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _huberDelta = huberDelta;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HuberCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        _robustCorrelations = new double[p];

        // Standardize y using robust estimators
        double yMedian = ComputeMedian(y);
        double yMAD = ComputeMAD(y, yMedian) + 1e-10;
        var yStd = y.Select(yi => (yi - yMedian) / yMAD).ToArray();

        for (int j = 0; j < p; j++)
        {
            // Standardize x using robust estimators
            var x = new double[n];
            for (int i = 0; i < n; i++) x[i] = X[i, j];

            double xMedian = ComputeMedian(x);
            double xMAD = ComputeMAD(x, xMedian) + 1e-10;
            var xStd = x.Select(xi => (xi - xMedian) / xMAD).ToArray();

            // Compute Huber-weighted correlation
            double sumWxy = 0, sumWx2 = 0, sumWy2 = 0, sumW = 0;
            for (int i = 0; i < n; i++)
            {
                double w = HuberWeight(xStd[i]) * HuberWeight(yStd[i]);
                sumWxy += w * xStd[i] * yStd[i];
                sumWx2 += w * xStd[i] * xStd[i];
                sumWy2 += w * yStd[i] * yStd[i];
                sumW += w;
            }

            double denom = Math.Sqrt(sumWx2 * sumWy2);
            _robustCorrelations[j] = denom > 1e-10 ? Math.Abs(sumWxy / denom) : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _robustCorrelations[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double HuberWeight(double z)
    {
        double absZ = Math.Abs(z);
        return absZ <= _huberDelta ? 1.0 : _huberDelta / absZ;
    }

    private double ComputeMedian(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;
        if (n == 0) return 0;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private double ComputeMAD(double[] values, double median)
    {
        var absDeviations = values.Select(v => Math.Abs(v - median)).ToArray();
        return ComputeMedian(absDeviations) * 1.4826; // Scale factor for consistency
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HuberCorrelationSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HuberCorrelationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HuberCorrelationSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
