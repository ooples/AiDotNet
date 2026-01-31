using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// Bhattacharyya Distance Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the Bhattacharyya distance between class
/// distributions, which measures the separability between classes.
/// </para>
/// <para><b>For Beginners:</b> The Bhattacharyya distance measures how much
/// two probability distributions overlap. Features with high Bhattacharyya
/// distance have class distributions that don't overlap much, making them
/// good for distinguishing between classes.
/// </para>
/// </remarks>
public class BhattacharyyaDistanceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _distanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DistanceScores => _distanceScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BhattacharyyaDistanceSelector(
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
            "BhattacharyyaDistanceSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        var classes = y.Distinct().OrderBy(c => c).ToList();
        int nClasses = classes.Count;

        _distanceScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double totalDist = 0;
            int pairCount = 0;

            // Compute pairwise Bhattacharyya distances
            for (int c1 = 0; c1 < nClasses; c1++)
            {
                for (int c2 = c1 + 1; c2 < nClasses; c2++)
                {
                    // Get feature values for each class
                    var vals1 = new List<double>();
                    var vals2 = new List<double>();

                    for (int i = 0; i < n; i++)
                    {
                        if (y[i] == classes[c1])
                            vals1.Add(X[i, j]);
                        else if (y[i] == classes[c2])
                            vals2.Add(X[i, j]);
                    }

                    if (vals1.Count == 0 || vals2.Count == 0) continue;

                    // Compute mean and variance for each class
                    double mean1 = vals1.Average();
                    double mean2 = vals2.Average();
                    double var1 = vals1.Sum(v => (v - mean1) * (v - mean1)) / (vals1.Count - 1 + 1e-10);
                    double var2 = vals2.Sum(v => (v - mean2) * (v - mean2)) / (vals2.Count - 1 + 1e-10);

                    var1 = Math.Max(var1, 1e-10);
                    var2 = Math.Max(var2, 1e-10);

                    // Bhattacharyya distance for Gaussian distributions
                    double varMean = (var1 + var2) / 2;
                    double term1 = 0.25 * Math.Log(0.25 * (var1 / var2 + var2 / var1 + 2));
                    double term2 = 0.25 * (mean1 - mean2) * (mean1 - mean2) / varMean;

                    double dist = term1 + term2;
                    totalDist += dist;
                    pairCount++;
                }
            }

            _distanceScores[j] = pairCount > 0 ? totalDist / pairCount : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _distanceScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BhattacharyyaDistanceSelector has not been fitted.");

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
        throw new NotSupportedException("BhattacharyyaDistanceSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BhattacharyyaDistanceSelector has not been fitted.");

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
