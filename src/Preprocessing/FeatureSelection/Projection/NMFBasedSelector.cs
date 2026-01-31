using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Non-negative Matrix Factorization (NMF) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses NMF to decompose the data matrix and selects features based on
/// their contribution to the learned basis vectors.
/// </para>
/// <para><b>For Beginners:</b> NMF breaks down your data into parts (like
/// breaking a photo into its component patterns). Features that contribute
/// most to these fundamental patterns are the most informative ones to keep.
/// NMF is especially good when features represent non-negative quantities.
/// </para>
/// </remarks>
public class NMFBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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

    public NMFBasedSelector(
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
            {
                double val = NumOps.ToDouble(data[i, j]);
                X[i, j] = Math.Max(0, val); // Ensure non-negative
            }

        int k = Math.Min(_nComponents, Math.Min(n, p));
        var rand = RandomHelper.CreateSecureRandom();

        // Initialize W (n x k) and H (k x p) with random non-negative values
        var W = new double[n, k];
        var H = new double[k, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                W[i, j] = rand.NextDouble() + 0.01;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < p; j++)
                H[i, j] = rand.NextDouble() + 0.01;

        // Multiplicative update rules
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Update H: H = H .* (W'X) ./ (W'WH + eps)
            var WtX = new double[k, p];
            var WtW = new double[k, k];
            var WtWH = new double[k, p];

            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < p; j++)
                    for (int r = 0; r < n; r++)
                        WtX[i, j] += W[r, i] * X[r, j];

                for (int j = 0; j < k; j++)
                    for (int r = 0; r < n; r++)
                        WtW[i, j] += W[r, i] * W[r, j];
            }

            for (int i = 0; i < k; i++)
                for (int j = 0; j < p; j++)
                    for (int r = 0; r < k; r++)
                        WtWH[i, j] += WtW[i, r] * H[r, j];

            for (int i = 0; i < k; i++)
                for (int j = 0; j < p; j++)
                    H[i, j] *= WtX[i, j] / (WtWH[i, j] + 1e-10);

            // Update W: W = W .* (XH') ./ (WHH' + eps)
            var XHt = new double[n, k];
            var HHt = new double[k, k];
            var WHHt = new double[n, k];

            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                    for (int r = 0; r < p; r++)
                        XHt[i, j] += X[i, r] * H[j, r];

            for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++)
                    for (int r = 0; r < p; r++)
                        HHt[i, j] += H[i, r] * H[j, r];

            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                    for (int r = 0; r < k; r++)
                        WHHt[i, j] += W[i, r] * HHt[r, j];

            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                    W[i, j] *= XHt[i, j] / (WHHt[i, j] + 1e-10);
        }

        // Feature scores = sum of squared H values across components
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            for (int c = 0; c < k; c++)
                _featureScores[j] += H[c, j] * H[c, j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NMFBasedSelector has not been fitted.");

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
        throw new NotSupportedException("NMFBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NMFBasedSelector has not been fitted.");

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
