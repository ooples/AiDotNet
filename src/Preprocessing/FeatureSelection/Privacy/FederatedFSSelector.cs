using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Privacy;

/// <summary>
/// Federated Feature Selection Selector.
/// </summary>
/// <remarks>
/// <para>
/// Simulates federated feature selection where data is partitioned across
/// multiple clients, and feature importance is computed locally then aggregated.
/// </para>
/// <para><b>For Beginners:</b> Federated learning keeps data on local devices/servers
/// (clients) and only shares model updates. This selector simulates that by:
/// 1) Splitting data into partitions (simulating different clients)
/// 2) Computing feature importance locally on each partition
/// 3) Aggregating scores across clients without sharing raw data
/// This provides privacy as raw data never leaves its partition.
/// </para>
/// </remarks>
public class FederatedFSSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClients;
    private readonly string _aggregationMethod;

    private double[]? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NClients => _nClients;
    public string AggregationMethod => _aggregationMethod;
    public double[]? AggregatedScores => _aggregatedScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FederatedFSSelector(
        int nFeaturesToSelect = 10,
        int nClients = 5,
        string aggregationMethod = "weighted_average",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nClients < 2)
            throw new ArgumentException("Number of clients must be at least 2.", nameof(nClients));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClients = nClients;
        _aggregationMethod = aggregationMethod;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FederatedFSSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Partition data across clients
        int baseSize = n / _nClients;
        var clientScores = new double[_nClients][];
        var clientSizes = new int[_nClients];

        for (int c = 0; c < _nClients; c++)
        {
            int start = c * baseSize;
            int end = (c == _nClients - 1) ? n : (c + 1) * baseSize;
            int clientN = end - start;
            clientSizes[c] = clientN;

            // Compute local feature scores using correlation
            clientScores[c] = ComputeLocalScores(X, y, start, end, p);
        }

        // Aggregate scores across clients
        _aggregatedScores = new double[p];

        switch (_aggregationMethod.ToLower())
        {
            case "weighted_average":
                // Weight by client data size
                double totalSize = clientSizes.Sum();
                for (int j = 0; j < p; j++)
                {
                    for (int c = 0; c < _nClients; c++)
                        _aggregatedScores[j] += clientScores[c][j] * clientSizes[c] / totalSize;
                }
                break;

            case "median":
                // Robust aggregation using median
                for (int j = 0; j < p; j++)
                {
                    var scores = new double[_nClients];
                    for (int c = 0; c < _nClients; c++)
                        scores[c] = clientScores[c][j];
                    Array.Sort(scores);
                    _aggregatedScores[j] = scores[_nClients / 2];
                }
                break;

            case "trimmed_mean":
                // Remove extreme values
                for (int j = 0; j < p; j++)
                {
                    var scores = new double[_nClients];
                    for (int c = 0; c < _nClients; c++)
                        scores[c] = clientScores[c][j];
                    Array.Sort(scores);

                    // Trim top and bottom
                    int trimCount = Math.Max(1, _nClients / 5);
                    double sum = 0;
                    int count = 0;
                    for (int c = trimCount; c < _nClients - trimCount; c++)
                    {
                        sum += scores[c];
                        count++;
                    }
                    _aggregatedScores[j] = count > 0 ? sum / count : scores[_nClients / 2];
                }
                break;

            default:
                // Simple average
                for (int j = 0; j < p; j++)
                {
                    for (int c = 0; c < _nClients; c++)
                        _aggregatedScores[j] += clientScores[c][j] / _nClients;
                }
                break;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _aggregatedScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeLocalScores(double[,] X, double[] y, int start, int end, int p)
    {
        int localN = end - start;
        var scores = new double[p];

        // Compute target statistics locally
        double targetMean = 0;
        for (int i = start; i < end; i++)
            targetMean += y[i];
        targetMean /= localN;

        double targetVar = 0;
        for (int i = start; i < end; i++)
            targetVar += (y[i] - targetMean) * (y[i] - targetMean);

        for (int j = 0; j < p; j++)
        {
            // Compute feature mean
            double featureMean = 0;
            for (int i = start; i < end; i++)
                featureMean += X[i, j];
            featureMean /= localN;

            // Compute covariance and variance
            double cov = 0, varX = 0;
            for (int i = start; i < end; i++)
            {
                double dx = X[i, j] - featureMean;
                double dy = y[i] - targetMean;
                cov += dx * dy;
                varX += dx * dx;
            }

            double denom = Math.Sqrt(varX * targetVar);
            scores[j] = denom > 1e-10 ? Math.Abs(cov / denom) : 0;
        }

        return scores;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FederatedFSSelector has not been fitted.");

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
        throw new NotSupportedException("FederatedFSSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FederatedFSSelector has not been fitted.");

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
