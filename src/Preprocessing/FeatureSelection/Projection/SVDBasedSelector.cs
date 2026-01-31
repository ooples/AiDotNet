using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// SVD-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Singular Value Decomposition to identify features that contribute
/// most to the dominant singular vectors.
/// </para>
/// <para><b>For Beginners:</b> SVD breaks down your data into components ordered
/// by importance. Features that have high values in the most important components
/// are the features that capture the most information about the data's structure.
/// </para>
/// </remarks>
public class SVDBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SVDBasedSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
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

        // Power iteration for top k right singular vectors (V)
        var V = new double[p, k];
        for (int c = 0; c < k; c++)
        {
            // Initialize random vector
            var v = new double[p];
            for (int j = 0; j < p; j++) v[j] = rand.NextDouble() - 0.5;
            Normalize(v);

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
            {
                // Orthogonalize against previous components
                for (int prev = 0; prev < c; prev++)
                {
                    double dot = 0;
                    for (int j = 0; j < p; j++) dot += v[j] * V[j, prev];
                    for (int j = 0; j < p; j++) v[j] -= dot * V[j, prev];
                }

                // X'X * v
                var u = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < p; j++)
                        u[i] += X[i, j] * v[j];

                var newV = new double[p];
                for (int j = 0; j < p; j++)
                    for (int i = 0; i < n; i++)
                        newV[j] += X[i, j] * u[i];

                Normalize(newV);
                Array.Copy(newV, v, p);
            }

            for (int j = 0; j < p; j++)
                V[j, c] = v[j];
        }

        // Compute singular values
        var singularValues = new double[k];
        for (int c = 0; c < k; c++)
        {
            var u = new double[n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < p; j++)
                    u[i] += X[i, j] * V[j, c];
            singularValues[c] = Math.Sqrt(u.Sum(x => x * x));
        }

        // Feature scores = weighted sum of squared loadings
        _featureScores = new double[p];
        double totalVar = singularValues.Sum(s => s * s);
        for (int c = 0; c < k; c++)
        {
            double weight = singularValues[c] * singularValues[c] / (totalVar + 1e-10);
            for (int j = 0; j < p; j++)
                _featureScores[j] += weight * V[j, c] * V[j, c];
        }

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
            throw new InvalidOperationException("SVDBasedSelector has not been fitted.");

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
        throw new NotSupportedException("SVDBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SVDBasedSelector has not been fitted.");

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
