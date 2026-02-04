using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Redundancy;

/// <summary>
/// Minimal Redundancy Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that minimize redundancy among selected features while
/// maximizing relevance to the target (mRMR-like approach).
/// </para>
/// <para><b>For Beginners:</b> When features are highly correlated with each
/// other, they provide similar information. This selector picks features that
/// are different from each other while still being useful for prediction.
/// </para>
/// </remarks>
public class MinimalRedundancySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _mrmrScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MRMRScores => _mrmrScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MinimalRedundancySelector(
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
            "MinimalRedundancySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Compute relevance (correlation with target) for each feature
        var relevance = new double[p];
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];
            relevance[j] = Math.Abs(ComputeCorrelation(col, y));
        }

        // Compute feature-feature correlation matrix
        var correlation = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            correlation[j1, j1] = 1;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                var col1 = new double[n];
                var col2 = new double[n];
                for (int i = 0; i < n; i++)
                {
                    col1[i] = X[i, j1];
                    col2[i] = X[i, j2];
                }
                double corr = Math.Abs(ComputeCorrelation(col1, col2));
                correlation[j1, j2] = corr;
                correlation[j2, j1] = corr;
            }
        }

        // Greedy mRMR selection
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();
        _mrmrScores = new double[p];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // First feature: highest relevance
        int firstFeature = Enumerable.Range(0, p).OrderByDescending(j => relevance[j]).First();
        selected.Add(firstFeature);
        remaining.Remove(firstFeature);
        _mrmrScores[firstFeature] = relevance[firstFeature];

        // Subsequent features: maximize relevance - redundancy
        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            double bestScore = double.MinValue;
            int bestFeature = -1;

            foreach (int j in remaining)
            {
                // Compute average redundancy with selected features
                double redundancy = selected.Average(s => correlation[j, s]);

                // mRMR score
                double score = relevance[j] - redundancy;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                _mrmrScores[bestFeature] = bestScore;
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
            throw new InvalidOperationException("MinimalRedundancySelector has not been fitted.");

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
        throw new NotSupportedException("MinimalRedundancySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MinimalRedundancySelector has not been fitted.");

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
