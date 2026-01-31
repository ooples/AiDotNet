using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Jackknife based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their stability under jackknife resampling,
/// which systematically leaves out one sample at a time.
/// </para>
/// <para><b>For Beginners:</b> Jackknife is a resampling method that removes one
/// data point at a time and recalculates statistics. Features with consistent
/// importance scores across all jackknife samples are more reliable.
/// </para>
/// </remarks>
public class JackknifeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxJackknifeIterations;

    private double[]? _meanImportance;
    private double[]? _importanceVariance;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MeanImportance => _meanImportance;
    public double[]? ImportanceVariance => _importanceVariance;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public JackknifeSelector(
        int nFeaturesToSelect = 10,
        int maxJackknifeIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxJackknifeIterations = maxJackknifeIterations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "JackknifeSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        int nIterations = Math.Min(n, _maxJackknifeIterations);
        var allImportance = new double[nIterations, p];

        // Jackknife: leave one out at a time
        for (int leave = 0; leave < nIterations; leave++)
        {
            // Extract leave-one-out data
            var subX = new double[n - 1, p];
            var subY = new double[n - 1];
            int idx = 0;
            for (int i = 0; i < n; i++)
            {
                if (i == leave) continue;
                subY[idx] = y[i];
                for (int j = 0; j < p; j++)
                    subX[idx, j] = X[i, j];
                idx++;
            }

            // Compute feature importance (correlation with target)
            for (int j = 0; j < p; j++)
            {
                var col = new double[n - 1];
                for (int i = 0; i < n - 1; i++)
                    col[i] = subX[i, j];
                allImportance[leave, j] = Math.Abs(ComputeCorrelation(col, subY));
            }
        }

        // Compute mean and variance of importance across jackknife samples
        _meanImportance = new double[p];
        _importanceVariance = new double[p];

        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int it = 0; it < nIterations; it++)
                sum += allImportance[it, j];
            _meanImportance[j] = sum / nIterations;

            double varSum = 0;
            for (int it = 0; it < nIterations; it++)
            {
                double diff = allImportance[it, j] - _meanImportance[j];
                varSum += diff * diff;
            }
            _importanceVariance[j] = varSum / nIterations;
        }

        // Select features with high mean importance and low variance (stable)
        var stabilityScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            // Higher mean and lower variance is better
            double stabilityPenalty = Math.Sqrt(_importanceVariance[j]) + 1e-6;
            stabilityScores[j] = _meanImportance[j] / stabilityPenalty;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => stabilityScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
        if (n < 2) return 0;

        double xMean = x.Average();
        double yMean = y.Average();

        double numerator = 0, xSumSq = 0, ySumSq = 0;
        for (int i = 0; i < n; i++)
        {
            numerator += (x[i] - xMean) * (y[i] - yMean);
            xSumSq += (x[i] - xMean) * (x[i] - xMean);
            ySumSq += (y[i] - yMean) * (y[i] - yMean);
        }

        double denominator = Math.Sqrt(xSumSq * ySumSq);
        return denominator > 1e-10 ? numerator / denominator : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JackknifeSelector has not been fitted.");

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
        throw new NotSupportedException("JackknifeSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JackknifeSelector has not been fitted.");

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
