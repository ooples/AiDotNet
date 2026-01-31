using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Correlation-based Feature Selection (CFS) for selecting feature subsets.
/// </summary>
/// <remarks>
/// <para>
/// CFS evaluates feature subsets based on a heuristic: good feature sets contain
/// features highly correlated with the class but uncorrelated with each other.
/// Uses greedy forward selection with the CFS merit function.
/// </para>
/// <para><b>For Beginners:</b> CFS looks for features that are good at predicting
/// the target but aren't redundant with each other. If two features give the same
/// information, CFS keeps only one. This prevents selecting features that are
/// copies of each other while maximizing predictive power.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _featureClassCorrelations;
    private double[,]? _featureInterCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureClassCorrelations => _featureClassCorrelations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CFS(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute feature-class correlations
        _featureClassCorrelations = new double[p];
        var targetArray = new double[n];
        for (int i = 0; i < n; i++)
            targetArray[i] = NumOps.ToDouble(target[i]);

        for (int j = 0; j < p; j++)
        {
            var featureArray = new double[n];
            for (int i = 0; i < n; i++)
                featureArray[i] = NumOps.ToDouble(data[i, j]);

            _featureClassCorrelations[j] = Math.Abs(ComputeCorrelation(featureArray, targetArray));
        }

        // Compute feature-feature correlations
        _featureInterCorrelations = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            var feature1 = new double[n];
            for (int i = 0; i < n; i++)
                feature1[i] = NumOps.ToDouble(data[i, j1]);

            for (int j2 = j1; j2 < p; j2++)
            {
                if (j1 == j2)
                {
                    _featureInterCorrelations[j1, j2] = 1.0;
                }
                else
                {
                    var feature2 = new double[n];
                    for (int i = 0; i < n; i++)
                        feature2[i] = NumOps.ToDouble(data[i, j2]);

                    double corr = Math.Abs(ComputeCorrelation(feature1, feature2));
                    _featureInterCorrelations[j1, j2] = corr;
                    _featureInterCorrelations[j2, j1] = corr;
                }
            }
        }

        // Greedy forward selection using CFS merit
        var selected = new List<int>();
        var available = Enumerable.Range(0, p).ToHashSet();

        while (selected.Count < _nFeaturesToSelect && available.Count > 0)
        {
            int bestFeature = -1;
            double bestMerit = double.NegativeInfinity;

            foreach (int j in available)
            {
                var candidate = new List<int>(selected) { j };
                double merit = ComputeCFSMerit(candidate);

                if (merit > bestMerit)
                {
                    bestMerit = merit;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                available.Remove(bestFeature);
            }
            else
            {
                break;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double ComputeCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
        double xMean = x.Average();
        double yMean = y.Average();

        double covariance = 0, xVar = 0, yVar = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = x[i] - xMean;
            double yDiff = y[i] - yMean;
            covariance += xDiff * yDiff;
            xVar += xDiff * xDiff;
            yVar += yDiff * yDiff;
        }

        double denom = Math.Sqrt(xVar * yVar);
        return denom > 1e-10 ? covariance / denom : 0;
    }

    private double ComputeCFSMerit(List<int> subset)
    {
        if (subset.Count == 0) return 0;

        int k = subset.Count;

        // Average feature-class correlation
        double avgFCCorr = subset.Sum(j => _featureClassCorrelations![j]) / k;

        // Average feature-feature correlation
        double avgFFCorr = 0;
        int pairs = 0;
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                avgFFCorr += _featureInterCorrelations![subset[i], subset[j]];
                pairs++;
            }
        }
        if (pairs > 0) avgFFCorr /= pairs;

        // CFS merit function: k * avgFCCorr / sqrt(k + k*(k-1)*avgFFCorr)
        double denom = Math.Sqrt(k + k * (k - 1) * avgFFCorr);
        return denom > 1e-10 ? k * avgFCCorr / denom : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CFS has not been fitted.");

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
        throw new NotSupportedException("CFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
