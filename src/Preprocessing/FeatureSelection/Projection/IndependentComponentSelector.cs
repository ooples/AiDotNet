using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Independent Component Analysis based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to independent components,
/// identifying features that contribute to statistically independent signals.
/// </para>
/// <para><b>For Beginners:</b> ICA separates mixed signals into independent sources
/// (like separating individual voices from a recording). This selector finds
/// features that contribute most to these independent signals.
/// </para>
/// </remarks>
public class IndependentComponentSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;
    private readonly int? _randomState;

    private double[]? _independenceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NComponents => _nComponents;
    public double[]? IndependenceScores => _independenceScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public IndependentComponentSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Center the data
        var means = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] -= means[j];

        // Whitening via PCA
        var covariance = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += X[i, j1] * X[i, j2];
                covariance[j1, j2] = sum / n;
                covariance[j2, j1] = covariance[j1, j2];
            }
        }

        // Simplified ICA using kurtosis as non-Gaussianity measure
        int numComponents = Math.Min(_nComponents, p);
        var mixingMatrix = new double[p, numComponents];

        // Initialize random mixing matrix
        for (int j = 0; j < p; j++)
            for (int c = 0; c < numComponents; c++)
                mixingMatrix[j, c] = rand.NextDouble() - 0.5;

        // Normalize columns
        for (int c = 0; c < numComponents; c++)
        {
            double norm = 0;
            for (int j = 0; j < p; j++)
                norm += mixingMatrix[j, c] * mixingMatrix[j, c];
            norm = Math.Sqrt(norm);
            if (norm > 1e-10)
                for (int j = 0; j < p; j++)
                    mixingMatrix[j, c] /= norm;
        }

        // FastICA iterations
        for (int c = 0; c < numComponents; c++)
        {
            var w = new double[p];
            for (int j = 0; j < p; j++)
                w[j] = mixingMatrix[j, c];

            for (int iter = 0; iter < 100; iter++)
            {
                // Compute w^T * X
                var wtx = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < p; j++)
                        wtx[i] += w[j] * X[i, j];

                // Use tanh non-linearity (approximation to negentropy)
                var gwtx = wtx.Select(x => Math.Tanh(x)).ToArray();
                var gpwtx = wtx.Select(x => 1 - Math.Tanh(x) * Math.Tanh(x)).ToArray();

                // Update rule: w = E{X * g(w^T * X)} - E{g'(w^T * X)} * w
                var newW = new double[p];
                for (int j = 0; j < p; j++)
                {
                    double term1 = 0;
                    for (int i = 0; i < n; i++)
                        term1 += X[i, j] * gwtx[i];
                    term1 /= n;

                    double term2 = gpwtx.Average() * w[j];
                    newW[j] = term1 - term2;
                }

                // Gram-Schmidt orthogonalization against previous components
                for (int prev = 0; prev < c; prev++)
                {
                    double dot = 0;
                    for (int j = 0; j < p; j++)
                        dot += newW[j] * mixingMatrix[j, prev];
                    for (int j = 0; j < p; j++)
                        newW[j] -= dot * mixingMatrix[j, prev];
                }

                // Normalize
                double norm = Math.Sqrt(newW.Sum(x => x * x));
                if (norm > 1e-10)
                    for (int j = 0; j < p; j++)
                        newW[j] /= norm;

                // Check convergence
                double diff = 0;
                for (int j = 0; j < p; j++)
                    diff += Math.Abs(Math.Abs(newW[j]) - Math.Abs(w[j]));

                w = newW;
                if (diff < 1e-6) break;
            }

            for (int j = 0; j < p; j++)
                mixingMatrix[j, c] = w[j];
        }

        // Score features by their total contribution to independent components
        _independenceScores = new double[p];
        for (int j = 0; j < p; j++)
            for (int c = 0; c < numComponents; c++)
                _independenceScores[j] += Math.Abs(mixingMatrix[j, c]);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _independenceScores[j])
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
            throw new InvalidOperationException("IndependentComponentSelector has not been fitted.");

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
        throw new NotSupportedException("IndependentComponentSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IndependentComponentSelector has not been fitted.");

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
