using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Bayesian Network structure-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their connections in a learned Bayesian network
/// structure, identifying features directly connected to the target.
/// </para>
/// <para><b>For Beginners:</b> A Bayesian network shows how variables depend on
/// each other. This selector finds features that have direct probabilistic
/// relationships with the target - these are the features that matter most
/// for prediction.
/// </para>
/// </remarks>
public class BayesianNetworkSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxParents;

    private double[]? _connectionStrengths;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxParents => _maxParents;
    public double[]? ConnectionStrengths => _connectionStrengths;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BayesianNetworkSelector(
        int nFeaturesToSelect = 10,
        int maxParents = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxParents = maxParents;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BayesianNetworkSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _connectionStrengths = new double[p];

        // Compute mutual information between each feature and target
        var miWithTarget = new double[p];
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];
            miWithTarget[j] = ComputeMI(col, y, n);
        }

        // Compute conditional mutual information (Markov blanket approximation)
        for (int j = 0; j < p; j++)
        {
            double directMI = miWithTarget[j];

            // Check if feature j's information is redundant given other features
            double maxRedundancy = 0;
            for (int k = 0; k < p; k++)
            {
                if (k == j) continue;

                var colJ = new double[n];
                var colK = new double[n];
                for (int i = 0; i < n; i++)
                {
                    colJ[i] = X[i, j];
                    colK[i] = X[i, k];
                }

                double miJK = ComputeMI(colJ, colK, n);
                double redundancy = Math.Min(directMI, Math.Min(miWithTarget[k], miJK));
                maxRedundancy = Math.Max(maxRedundancy, redundancy);
            }

            // Connection strength = direct MI minus redundancy
            _connectionStrengths[j] = Math.Max(0, directMI - 0.5 * maxRedundancy);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _connectionStrengths[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeMI(double[] x, double[] y, int n)
    {
        int nBins = Math.Max(5, (int)Math.Sqrt(n / 5));

        double xMin = x.Min(), xMax = x.Max();
        double yMin = y.Min(), yMax = y.Max();

        if (Math.Abs(xMax - xMin) < 1e-10 || Math.Abs(yMax - yMin) < 1e-10)
            return 0;

        double xBinWidth = (xMax - xMin) / nBins + 1e-10;
        double yBinWidth = (yMax - yMin) / nBins + 1e-10;

        var jointCounts = new int[nBins, nBins];
        var xCounts = new int[nBins];
        var yCounts = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            int xBin = Math.Min((int)((x[i] - xMin) / xBinWidth), nBins - 1);
            int yBin = Math.Min((int)((y[i] - yMin) / yBinWidth), nBins - 1);

            jointCounts[xBin, yBin]++;
            xCounts[xBin]++;
            yCounts[yBin]++;
        }

        double mi = 0;
        for (int xb = 0; xb < nBins; xb++)
        {
            for (int yb = 0; yb < nBins; yb++)
            {
                if (jointCounts[xb, yb] > 0 && xCounts[xb] > 0 && yCounts[yb] > 0)
                {
                    double pxy = (double)jointCounts[xb, yb] / n;
                    double px = (double)xCounts[xb] / n;
                    double py = (double)yCounts[yb] / n;
                    mi += pxy * Math.Log(pxy / (px * py) + 1e-10) / Math.Log(2);
                }
            }
        }

        return Math.Max(0, mi);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianNetworkSelector has not been fitted.");

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
        throw new NotSupportedException("BayesianNetworkSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianNetworkSelector has not been fitted.");

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
