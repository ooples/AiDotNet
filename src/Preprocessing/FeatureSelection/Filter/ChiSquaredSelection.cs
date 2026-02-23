using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Chi-Squared Feature Selection for categorical features.
/// </summary>
/// <remarks>
/// <para>
/// Uses the chi-squared statistic to measure the dependency between each
/// feature and the target variable. Features with higher chi-squared values
/// have stronger associations with the target.
/// </para>
/// <para><b>For Beginners:</b> The chi-squared test checks if two things
/// are related. If knowing the value of a feature helps predict the target,
/// they're related (high chi-squared). If the feature's value tells you
/// nothing about the target, chi-squared is low. This method keeps features
/// that are most related to what you're trying to predict.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChiSquaredSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _chiSquaredScores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ChiSquaredScores => _chiSquaredScores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ChiSquaredSelection(
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ChiSquaredSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _chiSquaredScores = new double[p];
        _pValues = new double[p];

        // Discretize target
        var targetClasses = new Dictionary<int, int>();
        var targetBins = new int[n];
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!targetClasses.ContainsKey(label))
                targetClasses[label] = targetClasses.Count;
            targetBins[i] = targetClasses[label];
        }

        int nClasses = targetClasses.Count;
        var classCounts = new int[nClasses];
        foreach (int b in targetBins)
            classCounts[b]++;

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            var values = new double[n];
            double minVal = double.MaxValue, maxVal = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                values[i] = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, values[i]);
                maxVal = Math.Max(maxVal, values[i]);
            }

            var featureBins = new int[n];
            double range = maxVal - minVal;

            for (int i = 0; i < n; i++)
            {
                featureBins[i] = range > 1e-10
                    ? Math.Min((int)((values[i] - minVal) / range * (_nBins - 1)), _nBins - 1)
                    : 0;
            }

            var featureCounts = new int[_nBins];
            foreach (int b in featureBins)
                featureCounts[b]++;

            // Build contingency table
            var observed = new int[_nBins, nClasses];
            for (int i = 0; i < n; i++)
                observed[featureBins[i], targetBins[i]]++;

            // Compute chi-squared
            double chiSquared = 0;
            int df = 0;

            for (int b = 0; b < _nBins; b++)
            {
                if (featureCounts[b] == 0) continue;
                df++;

                for (int c = 0; c < nClasses; c++)
                {
                    if (classCounts[c] == 0) continue;

                    double expected = (double)featureCounts[b] * classCounts[c] / n;
                    if (expected > 0)
                    {
                        double diff = observed[b, c] - expected;
                        chiSquared += diff * diff / expected;
                    }
                }
            }

            _chiSquaredScores[j] = chiSquared;
            df = (df - 1) * (nClasses - 1);
            _pValues[j] = df > 0 ? ChiSquaredPValue(chiSquared, df) : 1.0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _chiSquaredScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ChiSquaredPValue(double chiSquared, int df)
    {
        // Approximation using Wilson-Hilferty transformation
        if (df <= 0) return 1.0;

        double z = Math.Pow(chiSquared / df, 1.0 / 3.0) - (1 - 2.0 / (9 * df));
        z /= Math.Sqrt(2.0 / (9 * df));

        // Standard normal CDF approximation
        return 1 - NormalCDF(z);
    }

    private double NormalCDF(double x)
    {
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x) / Math.Sqrt(2);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return 0.5 * (1.0 + sign * y);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquaredSelection has not been fitted.");

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
        throw new NotSupportedException("ChiSquaredSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquaredSelection has not been fitted.");

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
