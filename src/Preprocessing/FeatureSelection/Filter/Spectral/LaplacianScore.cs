using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Laplacian Score for unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Laplacian Score evaluates features based on their ability to preserve locality
/// in the data. Features with low Laplacian Score preserve local structure well,
/// meaning similar samples have similar feature values.
/// </para>
/// <para><b>For Beginners:</b> The Laplacian Score measures how smoothly a feature
/// varies across similar data points. A good feature should have similar values
/// for nearby points (like neighbors having similar house prices). Features that
/// jump around randomly between neighbors score poorly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LaplacianScore<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _kernelWidth;

    private double[]? _laplacianScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double[]? LaplacianScores => _laplacianScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LaplacianScore(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double kernelWidth = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _kernelWidth = kernelWidth;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Step 1: Build k-NN graph with heat kernel weights
        var weights = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            // Compute distances to all other points
            var distances = new List<(int Index, double Distance)>();
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;

                double dist = 0;
                for (int f = 0; f < p; f++)
                {
                    double diff = NumOps.ToDouble(data[i, f]) - NumOps.ToDouble(data[j, f]);
                    dist += diff * diff;
                }
                distances.Add((j, dist));
            }

            // Keep k nearest neighbors
            var neighbors = distances.OrderBy(x => x.Distance).Take(_nNeighbors);
            foreach (var (j, dist) in neighbors)
            {
                // Heat kernel weight
                double weight = Math.Exp(-dist / (_kernelWidth * _kernelWidth));
                weights[i, j] = weight;
                weights[j, i] = weight; // Symmetric
            }
        }

        // Step 2: Compute diagonal matrix D
        var D = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                D[i] += weights[i, j];
        }

        // Step 3: Compute Laplacian Score for each feature
        _laplacianScores = new double[p];

        for (int f = 0; f < p; f++)
        {
            // Get feature values
            var featureVals = new double[n];
            double sumD = 0;
            double sumFD = 0;

            for (int i = 0; i < n; i++)
            {
                featureVals[i] = NumOps.ToDouble(data[i, f]);
                sumD += D[i];
                sumFD += featureVals[i] * D[i];
            }

            // Center the feature
            double mean = sumFD / sumD;
            var centered = new double[n];
            for (int i = 0; i < n; i++)
                centered[i] = featureVals[i] - mean;

            // Numerator: f^T L f where L = D - W
            double numerator = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double laplacian = (i == j) ? D[i] - weights[i, j] : -weights[i, j];
                    numerator += centered[i] * laplacian * centered[j];
                }
            }

            // Denominator: f^T D f
            double denominator = 0;
            for (int i = 0; i < n; i++)
                denominator += centered[i] * D[i] * centered[i];

            _laplacianScores[f] = denominator > 1e-10 ? numerator / denominator : double.MaxValue;
        }

        // Select features with lowest Laplacian Score (better locality preservation)
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _laplacianScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderBy(x => x.Score) // Lower is better
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScore has not been fitted.");

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
        throw new NotSupportedException("LaplacianScore does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaplacianScore has not been fitted.");

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
