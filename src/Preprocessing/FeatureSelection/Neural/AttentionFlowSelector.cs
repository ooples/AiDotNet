using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Attention Flow based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using attention flow analysis, which tracks how
/// attention propagates through feature interactions.
/// </para>
/// <para><b>For Beginners:</b> Attention flow measures how "attention" spreads
/// from the output back to input features through a graph of feature interactions.
/// Features that receive more attention flow are considered more important.
/// This captures both direct and indirect feature contributions.
/// </para>
/// </remarks>
public class AttentionFlowSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;

    private double[]? _attentionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public double[]? AttentionScores => _attentionScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AttentionFlowSelector(
        int nFeaturesToSelect = 10,
        int nIterations = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AttentionFlowSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Build attention matrix from feature correlations
        var attentionMatrix = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = 0; j2 < p; j2++)
            {
                if (j1 == j2)
                {
                    attentionMatrix[j1, j2] = 1.0;
                    continue;
                }

                // Compute attention weight as correlation
                var col1 = new double[n];
                var col2 = new double[n];
                for (int i = 0; i < n; i++)
                {
                    col1[i] = X[i, j1];
                    col2[i] = X[i, j2];
                }

                double mean1 = col1.Average();
                double mean2 = col2.Average();
                double cov = 0, var1 = 0, var2 = 0;
                for (int i = 0; i < n; i++)
                {
                    cov += (col1[i] - mean1) * (col2[i] - mean2);
                    var1 += (col1[i] - mean1) * (col1[i] - mean1);
                    var2 += (col2[i] - mean2) * (col2[i] - mean2);
                }

                double denom = Math.Sqrt(var1 * var2);
                attentionMatrix[j1, j2] = denom > 1e-10 ? Math.Abs(cov / denom) : 0;
            }
        }

        // Softmax normalize rows
        for (int j1 = 0; j1 < p; j1++)
        {
            double maxVal = 0;
            for (int j2 = 0; j2 < p; j2++)
                maxVal = Math.Max(maxVal, attentionMatrix[j1, j2]);

            double expSum = 0;
            for (int j2 = 0; j2 < p; j2++)
                expSum += Math.Exp(attentionMatrix[j1, j2] - maxVal);

            for (int j2 = 0; j2 < p; j2++)
                attentionMatrix[j1, j2] = Math.Exp(attentionMatrix[j1, j2] - maxVal) / expSum;
        }

        // Initial attention from target correlation
        var attention = new double[p];
        double targetMean = y.Average();
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double mean = col.Average();
            double cov = 0, varX = 0, varY = 0;
            for (int i = 0; i < n; i++)
            {
                cov += (col[i] - mean) * (y[i] - targetMean);
                varX += (col[i] - mean) * (col[i] - mean);
                varY += (y[i] - targetMean) * (y[i] - targetMean);
            }

            double denom = Math.Sqrt(varX * varY);
            attention[j] = denom > 1e-10 ? Math.Abs(cov / denom) : 0;
        }

        // Propagate attention flow
        for (int iter = 0; iter < _nIterations; iter++)
        {
            var newAttention = new double[p];
            for (int j1 = 0; j1 < p; j1++)
            {
                for (int j2 = 0; j2 < p; j2++)
                    newAttention[j1] += attentionMatrix[j2, j1] * attention[j2];
            }

            // Normalize
            double sum = newAttention.Sum();
            if (sum > 1e-10)
                for (int j = 0; j < p; j++)
                    newAttention[j] /= sum;

            attention = newAttention;
        }

        _attentionScores = attention;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _attentionScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AttentionFlowSelector has not been fitted.");

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
        throw new NotSupportedException("AttentionFlowSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AttentionFlowSelector has not been fitted.");

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
