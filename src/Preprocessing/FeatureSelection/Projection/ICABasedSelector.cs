using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Independent Component Analysis (ICA) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses ICA to find independent components and selects features that
/// contribute most to these statistically independent sources.
/// </para>
/// <para><b>For Beginners:</b> ICA finds hidden "sources" that combine to create
/// your observed data (like separating mixed audio signals). Features that
/// strongly contribute to these independent sources capture unique, non-redundant
/// information and are good candidates for selection.
/// </para>
/// </remarks>
public class ICABasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;
    private readonly int _nIterations;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ICABasedSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int nIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
        _nIterations = nIterations;
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

        // Center data
        var means = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) X[i, j] -= means[j];
        }

        int k = Math.Min(_nComponents, Math.Min(n, p));
        var rand = RandomHelper.CreateSecureRandom();

        // Whitening using simple covariance-based approach
        var cov = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
            for (int j2 = j1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += X[i, j1] * X[i, j2];
                cov[j1, j2] = sum / n;
                cov[j2, j1] = cov[j1, j2];
            }

        // FastICA algorithm (simplified)
        var W = new double[p, k];
        for (int c = 0; c < k; c++)
        {
            // Initialize weight vector
            var w = new double[p];
            for (int j = 0; j < p; j++) w[j] = rand.NextDouble() - 0.5;
            Normalize(w);

            for (int iter = 0; iter < _nIterations; iter++)
            {
                // Compute w'X for all samples
                var wx = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < p; j++)
                        wx[i] += w[j] * X[i, j];

                // Non-linearity g(u) = tanh(u), g'(u) = 1 - tanh(u)^2
                var gWx = new double[n];
                var gPrimeWx = new double[n];
                for (int i = 0; i < n; i++)
                {
                    gWx[i] = Math.Tanh(wx[i]);
                    gPrimeWx[i] = 1 - gWx[i] * gWx[i];
                }

                // Update: w = E[X * g(w'X)] - E[g'(w'X)] * w
                var newW = new double[p];
                double meanGPrime = gPrimeWx.Average();

                for (int j = 0; j < p; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < n; i++)
                        sum += X[i, j] * gWx[i];
                    newW[j] = sum / n - meanGPrime * w[j];
                }

                // Orthogonalize against previous components
                for (int prev = 0; prev < c; prev++)
                {
                    double dot = 0;
                    for (int j = 0; j < p; j++) dot += newW[j] * W[j, prev];
                    for (int j = 0; j < p; j++) newW[j] -= dot * W[j, prev];
                }

                Normalize(newW);
                Array.Copy(newW, w, p);
            }

            for (int j = 0; j < p; j++)
                W[j, c] = w[j];
        }

        // Feature scores = sum of absolute weights across components
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            for (int c = 0; c < k; c++)
                _featureScores[j] += Math.Abs(W[j, c]);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private void Normalize(double[] v)
    {
        double norm = Math.Sqrt(v.Sum(x => x * x)) + 1e-10;
        for (int i = 0; i < v.Length; i++)
            v[i] /= norm;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ICABasedSelector has not been fitted.");

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
        throw new NotSupportedException("ICABasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ICABasedSelector has not been fitted.");

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
