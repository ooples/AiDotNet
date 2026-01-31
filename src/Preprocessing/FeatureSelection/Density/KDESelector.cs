using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Density;

/// <summary>
/// Kernel Density Estimation (KDE) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how well their density distributions separate
/// between different classes.
/// </para>
/// <para><b>For Beginners:</b> KDE estimates how data points are spread out.
/// Features where different classes have clearly separate density patterns
/// are better for classification.
/// </para>
/// </remarks>
public class KDESelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _bandwidth;

    private double[]? _separationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Bandwidth => _bandwidth;
    public double[]? SeparationScores => _separationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KDESelector(
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
            "KDESelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _separationScores = new double[p];

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

            // Compute class-specific densities at evaluation points
            double minVal = col.Min();
            double maxVal = col.Max();
            int nEvalPoints = 50;
            var evalPoints = Enumerable.Range(0, nEvalPoints)
                .Select(i => minVal + (maxVal - minVal) * i / (nEvalPoints - 1))
                .ToArray();

            // Compute KDE for each class
            var classDensities = new Dictionary<int, double[]>();
            foreach (var c in classes)
            {
                var classData = classIndices[c].Select(i => col[i]).ToArray();
                classDensities[c] = ComputeKDE(classData, evalPoints);
            }

            // Compute separation score: average KL divergence between class densities
            double totalSeparation = 0;
            int pairs = 0;
            for (int ci = 0; ci < classes.Count; ci++)
            {
                for (int cj = ci + 1; cj < classes.Count; cj++)
                {
                    double kl = ComputeSymmetricKL(classDensities[classes[ci]], classDensities[classes[cj]]);
                    totalSeparation += kl;
                    pairs++;
                }
            }

            _separationScores[j] = pairs > 0 ? totalSeparation / pairs : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _separationScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeKDE(double[] data, double[] evalPoints)
    {
        var densities = new double[evalPoints.Length];
        double normFactor = 1.0 / (data.Length * _bandwidth * Math.Sqrt(2 * Math.PI));

        for (int i = 0; i < evalPoints.Length; i++)
        {
            double sum = 0;
            foreach (double x in data)
            {
                double u = (evalPoints[i] - x) / _bandwidth;
                sum += Math.Exp(-0.5 * u * u);
            }
            densities[i] = sum * normFactor + 1e-10; // Add small constant to avoid log(0)
        }

        return densities;
    }

    private double ComputeSymmetricKL(double[] p, double[] q)
    {
        double klPQ = 0, klQP = 0;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 1e-10 && q[i] > 1e-10)
            {
                klPQ += p[i] * Math.Log(p[i] / q[i]);
                klQP += q[i] * Math.Log(q[i] / p[i]);
            }
        }
        return (klPQ + klQP) / 2;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KDESelector has not been fitted.");

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
        throw new NotSupportedException("KDESelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KDESelector has not been fitted.");

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
