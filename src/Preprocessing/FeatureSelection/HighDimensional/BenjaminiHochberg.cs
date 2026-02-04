using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.HighDimensional;

/// <summary>
/// Benjamini-Hochberg procedure for feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) when
/// testing multiple hypotheses. It sorts p-values and finds the largest k such that
/// p_(k) <= k/m * alpha, where m is the number of tests.
/// </para>
/// <para><b>For Beginners:</b> When you test many features, some will look significant
/// just by chance (false positives). This method controls how many of your "discoveries"
/// are actually false. If you set FDR to 0.1, you expect at most 10% of selected
/// features to be false positives.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BenjaminiHochberg<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private double[]? _pValues;
    private double[]? _adjustedPValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double[]? PValues => _pValues;
    public double[]? AdjustedPValues => _adjustedPValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BenjaminiHochberg(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha <= 0 || alpha > 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BenjaminiHochberg requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Step 1: Compute p-values for each feature using correlation t-test
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute correlation
            double xMean = 0, yMean = 0;
            for (int i = 0; i < n; i++)
            {
                xMean += NumOps.ToDouble(data[i, j]);
                yMean += NumOps.ToDouble(target[i]);
            }
            xMean /= n;
            yMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            double r = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;

            // Convert correlation to t-statistic and then to p-value
            double t = r * Math.Sqrt((n - 2) / (1 - r * r + 1e-10));
            double df = n - 2;

            // Approximate p-value using normal distribution for large n
            _pValues[j] = 2 * (1 - NormalCDF(Math.Abs(t)));
        }

        // Step 2: Apply Benjamini-Hochberg procedure
        var sortedPValues = _pValues
            .Select((pval, idx) => (PValue: pval, Index: idx))
            .OrderBy(x => x.PValue)
            .ToList();

        _adjustedPValues = new double[p];

        // Compute adjusted p-values (BH adjusted)
        double[] qValues = new double[p];
        double minAdjusted = 1.0;
        for (int k = p - 1; k >= 0; k--)
        {
            int originalIdx = sortedPValues[k].Index;
            double adjusted = Math.Min(1.0, sortedPValues[k].PValue * p / (k + 1));
            adjusted = Math.Min(minAdjusted, adjusted);
            minAdjusted = adjusted;
            _adjustedPValues[originalIdx] = adjusted;
            qValues[k] = adjusted;
        }

        // Step 3: Select features with adjusted p-value <= alpha
        var selected = new List<int>();
        for (int k = 0; k < sortedPValues.Count; k++)
        {
            if (qValues[k] <= _alpha)
                selected.Add(sortedPValues[k].Index);
        }

        // If not enough selected, take top features by p-value
        if (selected.Count == 0)
        {
            _selectedIndices = sortedPValues
                .Take(Math.Min(_nFeaturesToSelect, p))
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = selected
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double NormalCDF(double x)
    {
        // Approximation of standard normal CDF
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BenjaminiHochberg has not been fitted.");

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
        throw new NotSupportedException("BenjaminiHochberg does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BenjaminiHochberg has not been fitted.");

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
