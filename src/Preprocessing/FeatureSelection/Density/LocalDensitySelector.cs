using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Density;

/// <summary>
/// Local Density based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to local density patterns,
/// identifying features that create meaningful local neighborhoods.
/// </para>
/// <para><b>For Beginners:</b> Local density measures how crowded the area around
/// each data point is. Features that create clear differences in local density
/// between classes help separate the data better.
/// </para>
/// </remarks>
public class LocalDensitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private double[]? _densityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double[]? DensityScores => _densityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LocalDensitySelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LocalDensitySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _densityScores = new double[p];
        int k = Math.Min(_nNeighbors, n - 1);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Standardize feature
            double mean = col.Average();
            double std = Math.Sqrt(col.Select(v => (v - mean) * (v - mean)).Average());
            if (std > 1e-10)
                for (int i = 0; i < n; i++)
                    col[i] = (col[i] - mean) / std;

            // Compute local density for each point (1/average distance to k nearest neighbors)
            var localDensity = new double[n];
            for (int i = 0; i < n; i++)
            {
                var distances = new List<double>();
                for (int i2 = 0; i2 < n; i2++)
                {
                    if (i2 != i)
                        distances.Add(Math.Abs(col[i] - col[i2]));
                }
                distances.Sort();

                double avgDist = distances.Take(k).Average();
                localDensity[i] = 1.0 / (avgDist + 1e-10);
            }

            // Compute correlation between local density and class label
            double[] classValues = y.Select(c => (double)c).ToArray();
            _densityScores[j] = Math.Abs(ComputeCorrelation(localDensity, classValues));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _densityScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
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
            throw new InvalidOperationException("LocalDensitySelector has not been fitted.");

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
        throw new NotSupportedException("LocalDensitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LocalDensitySelector has not been fitted.");

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
