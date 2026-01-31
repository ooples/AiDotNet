using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Factor Analysis based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their factor loadings, identifying features that
/// load heavily on extracted latent factors.
/// </para>
/// <para><b>For Beginners:</b> Factor analysis finds hidden patterns in data by
/// identifying underlying factors. Features that are strongly connected to these
/// factors are selected, as they carry the most meaningful information.
/// </para>
/// </remarks>
public class FactorAnalysisSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nFactors;

    private double[]? _communalities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NFactors => _nFactors;
    public double[]? Communalities => _communalities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FactorAnalysisSelector(
        int nFeaturesToSelect = 10,
        int nFactors = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nFactors = nFactors;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Standardize the data
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1));
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];

        // Compute correlation matrix
        var correlation = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            correlation[j1, j1] = 1;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += X[i, j1] * X[i, j2];
                correlation[j1, j2] = sum / (n - 1);
                correlation[j2, j1] = correlation[j1, j2];
            }
        }

        // Principal axis factoring (simplified)
        int numFactors = Math.Min(_nFactors, p);
        var loadings = new double[p, numFactors];

        // Initial communality estimates (use squared multiple correlations approx)
        var communalities = new double[p];
        for (int j = 0; j < p; j++)
        {
            double maxCorr = 0;
            for (int k = 0; k < p; k++)
                if (k != j)
                    maxCorr = Math.Max(maxCorr, Math.Abs(correlation[j, k]));
            communalities[j] = maxCorr * maxCorr;
        }

        // Iterative factor extraction
        var reducedCorr = (double[,])correlation.Clone();
        for (int f = 0; f < numFactors; f++)
        {
            // Set diagonal to communalities
            for (int j = 0; j < p; j++)
                reducedCorr[j, j] = communalities[j];

            // Extract factor via power iteration
            var factor = new double[p];
            for (int j = 0; j < p; j++)
                factor[j] = 1.0 / Math.Sqrt(p);

            for (int iter = 0; iter < 50; iter++)
            {
                var newFactor = new double[p];
                for (int j = 0; j < p; j++)
                    for (int k = 0; k < p; k++)
                        newFactor[j] += reducedCorr[j, k] * factor[k];

                double norm = Math.Sqrt(newFactor.Sum(x => x * x));
                if (norm > 1e-10)
                    for (int j = 0; j < p; j++)
                        newFactor[j] /= norm;

                factor = newFactor;
            }

            // Compute eigenvalue
            double eigenvalue = 0;
            for (int j = 0; j < p; j++)
            {
                double mv = 0;
                for (int k = 0; k < p; k++)
                    mv += reducedCorr[j, k] * factor[k];
                eigenvalue += mv * factor[j];
            }

            // Store loadings
            for (int j = 0; j < p; j++)
                loadings[j, f] = Math.Sqrt(Math.Max(0, eigenvalue)) * factor[j];

            // Deflate
            for (int j1 = 0; j1 < p; j1++)
                for (int j2 = 0; j2 < p; j2++)
                    reducedCorr[j1, j2] -= loadings[j1, f] * loadings[j2, f];
        }

        // Compute communalities (sum of squared loadings)
        _communalities = new double[p];
        for (int j = 0; j < p; j++)
            for (int f = 0; f < numFactors; f++)
                _communalities[j] += loadings[j, f] * loadings[j, f];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _communalities[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FactorAnalysisSelector has not been fitted.");

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
        throw new NotSupportedException("FactorAnalysisSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FactorAnalysisSelector has not been fitted.");

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
