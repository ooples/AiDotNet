using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Density;

/// <summary>
/// Kernel Density Estimation (KDE) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses kernel density estimation to select features that maximize the
/// separation of class-conditional density estimates.
/// </para>
/// <para><b>For Beginners:</b> KDE estimates the probability distribution of
/// your data smoothly. This selector finds features where different classes
/// have clearly different distributions, making it easier to tell classes apart.
/// </para>
/// </remarks>
public class KDEBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _bandwidth;

    private double[]? _kdeSeparationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Bandwidth => _bandwidth;
    public double[]? KDESeparationScores => _kdeSeparationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KDEBasedSelector(
        int nFeaturesToSelect = 10,
        double bandwidth = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _bandwidth = bandwidth;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KDEBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var classIndices = new Dictionary<int, List<int>>();
        foreach (var c in classes)
            classIndices[c] = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();

        _kdeSeparationScores = new double[p];

        // For each feature, compute KDE separation between classes
        for (int j = 0; j < p; j++)
        {
            // Get feature values per class
            var classValues = new Dictionary<int, double[]>();
            foreach (var c in classes)
                classValues[c] = classIndices[c].Select(i => X[i, j]).ToArray();

            // Compute Hellinger distance between class densities
            double totalSeparation = 0;
            int pairCount = 0;

            for (int c1Idx = 0; c1Idx < classes.Count; c1Idx++)
            {
                for (int c2Idx = c1Idx + 1; c2Idx < classes.Count; c2Idx++)
                {
                    int c1 = classes[c1Idx];
                    int c2 = classes[c2Idx];

                    double hellinger = ComputeHellingerDistance(
                        classValues[c1], classValues[c2], _bandwidth);
                    totalSeparation += hellinger;
                    pairCount++;
                }
            }

            _kdeSeparationScores[j] = pairCount > 0 ? totalSeparation / pairCount : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _kdeSeparationScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeHellingerDistance(double[] values1, double[] values2, double h)
    {
        // Determine evaluation points
        double min1 = values1.Min(), max1 = values1.Max();
        double min2 = values2.Min(), max2 = values2.Max();
        double minVal = Math.Min(min1, min2) - 3 * h;
        double maxVal = Math.Max(max1, max2) + 3 * h;

        int nPoints = 100;
        double step = (maxVal - minVal) / (nPoints - 1);

        double hellingerSum = 0;
        for (int i = 0; i < nPoints; i++)
        {
            double x = minVal + i * step;
            double density1 = ComputeKDE(x, values1, h);
            double density2 = ComputeKDE(x, values2, h);
            hellingerSum += (Math.Sqrt(density1) - Math.Sqrt(density2)) *
                           (Math.Sqrt(density1) - Math.Sqrt(density2));
        }

        return Math.Sqrt(hellingerSum * step / 2);
    }

    private double ComputeKDE(double x, double[] values, double h)
    {
        double sum = 0;
        foreach (double xi in values)
        {
            double u = (x - xi) / h;
            sum += Math.Exp(-0.5 * u * u) / Math.Sqrt(2 * Math.PI);
        }
        return sum / (values.Length * h + 1e-10);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KDEBasedSelector has not been fitted.");

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
        throw new NotSupportedException("KDEBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KDEBasedSelector has not been fitted.");

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
